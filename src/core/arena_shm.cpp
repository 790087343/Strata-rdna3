// src/core/arena_shm.cpp - [claude] STRATA_ARENA_SHM: the expert arena kept in shared memory across engine starts
// (see include/strata/core/arena_shm.hpp for the contract).
#include "strata/core/arena_shm.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)
#endif
#endif

namespace strata::core {

std::string ArenaShm::env_name() {
    const char* e = std::getenv("STRATA_ARENA_SHM");
    return e == nullptr ? std::string() : std::string(e);
}

#if !defined(__linux__)

ArenaShm::~ArenaShm() = default;
bool ArenaShm::open(const std::string&, uint64_t, uint64_t, const std::string&, std::string& note) {
    note = "STRATA_ARENA_SHM is Linux only";
    return false;
}
bool ArenaShm::mark_complete(std::string&) { return false; }
bool ArenaShm::drop_if_unused() { return false; }
void ArenaShm::discard_created() {}

#else

namespace {

constexpr uint64_t kHdrBytes = 2ull << 20;      // the arena starts 2 MiB in: its data keeps the 2 MiB alignment THP needs
constexpr uint64_t kGiB = 1ull << 30;
constexpr char kMagic[16] = "STRATA-ARENA-SH";
constexpr uint32_t kVersion = 1;
constexpr uint32_t kStateFilling = 0, kStateComplete = 1;

struct Header {
    char magic[16];
    uint32_t version;
    uint32_t state;
    uint64_t header_bytes;
    uint64_t arena_bytes;
    uint64_t key;
    uint64_t created_unix;
    int64_t creator_pid;
    char desc[1024];
};
static_assert(sizeof(Header) < kHdrBytes);

std::string hex16(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

std::string shm_dir() {
    const char* e = std::getenv("STRATA_ARENA_SHM_DIR");
    return e != nullptr && *e != '\0' ? std::string(e) : std::string("/dev/shm");
}

bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 64) return false;
    for (char c : n)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return false;
    return n != "." && n != "..";
}

bool pid_alive(pid_t p) { return p > 0 && (kill(p, 0) == 0 || errno == EPERM); }

// the keeper: a `sleep infinity` that holds the memfd.  Only that program, so a recycled pid is never signalled.
bool is_keeper(pid_t p) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/cmdline", (int) p);
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    char buf[256] = {0};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    std::vector<std::string> args;
    for (size_t i = 0; i < n;) {
        const size_t z = std::strlen(buf + i);
        args.emplace_back(buf + i, z);
        i += z + 1;
    }
    if (args.size() != 2 || args[1] != "infinity") return false;
    const size_t slash = args[0].find_last_of('/');
    return args[0].substr(slash == std::string::npos ? 0 : slash + 1) == "sleep";
}

bool read_keeper_index(const std::string& index, pid_t& pid, int& fdn) {
    std::FILE* f = std::fopen(index.c_str(), "r");
    if (f == nullptr) return false;
    int p = 0, n = -1;
    const bool ok = std::fscanf(f, "%d %d", &p, &n) == 2 && p > 0 && n >= 0;
    std::fclose(f);
    pid = (pid_t) p;
    fdn = n;
    return ok;
}

// the keeper's memfd, opened again (a new open file description: our flock is not the keeper's)
int open_keeper_fd(pid_t pid, int fdn) {
    if (!is_keeper(pid)) return -1;
    char link[64], target[256] = {0};
    std::snprintf(link, sizeof link, "/proc/%d/fd/%d", (int) pid, fdn);
    const ssize_t n = readlink(link, target, sizeof target - 1);
    if (n <= 0 || std::strncmp(target, "/memfd:strata-arena-", 20) != 0) return -1;
    return ::open(link, O_RDWR | O_CLOEXEC);
}

// what an object holds: -1 not one of ours / the wrong size or key, else its state
int header_state(int fd, uint64_t bytes, uint64_t key) {
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size != kHdrBytes + bytes) return -1;
    Header h{};
    if (pread(fd, &h, sizeof h, 0) != (ssize_t) sizeof h) return -1;
    if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kVersion || h.header_bytes != kHdrBytes ||
        h.arena_bytes != bytes || h.key != key)
        return -1;
    return (int) h.state;
}

// LOCK_SH, waiting (at most ~10 minutes) while a creator fills the object
bool lock_shared_wait(int fd, const std::string& what) {
    bool said = false;
    for (int i = 0; i < 2400; ++i) {
        if (flock(fd, LOCK_SH | LOCK_NB) == 0) return true;
        if (errno != EWOULDBLOCK && errno != EINTR) return false;
        if (!said) {
            std::fprintf(stderr, "strata generate: STRATA_ARENA_SHM: waiting for another engine to finish filling %s\n",
                         what.c_str());
            said = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return false;
}

// fallocate (tmpfs: zero-filled pages now, so a full /dev/shm is an error here rather than a SIGBUS in the load) and
// MADV_COLLAPSE (2 MiB folios: tmpfs never gives them on fault while shmem_enabled is "never", but collapse ignores that
// setting, and the folios stay huge in the object for every later start), both in 1 GiB pieces on several threads
template <class F>
void parallel_pieces(uint64_t off, uint64_t bytes, F&& f) {
    const uint64_t n = (bytes + kGiB - 1) / kGiB;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = (unsigned) std::min<uint64_t>(n, std::min(16u, hw));
    std::atomic<uint64_t> next{0};
    auto work = [&] {
        for (;;) {
            const uint64_t i = next.fetch_add(1);
            if (i >= n) break;
            const uint64_t a = off + i * kGiB, len = std::min<uint64_t>(kGiB, off + bytes - a);
            f(a, len);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
}

bool allocate_all(int fd, uint64_t file_bytes, std::string& err) {
    std::atomic<int> bad{0};
    parallel_pieces(0, file_bytes, [&](uint64_t a, uint64_t len) {
        if (bad.load()) return;
        if (fallocate(fd, 0, (off_t) a, (off_t) len) != 0) {
            int z = 0;
            bad.compare_exchange_strong(z, errno);
        }
    });
    if (bad.load() != 0) {
        err = std::string("fallocate: ") + std::strerror(bad.load());
        return false;
    }
    return true;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// delete one object of `name` that is not ours (another key) when no engine holds it; true when removed
bool remove_unused(const std::string& dir, const std::string& entry, uint64_t& freed) {
    const std::string full = dir + "/" + entry;
    const std::string keeper_suffix = ".keeper";
    if (entry.size() > keeper_suffix.size() &&
        entry.compare(entry.size() - keeper_suffix.size(), keeper_suffix.size(), keeper_suffix) == 0) {
        pid_t pid = 0;
        int fdn = -1;
        if (!read_keeper_index(full, pid, fdn) || !pid_alive(pid) || !is_keeper(pid)) {
            unlink(full.c_str());   // its keeper is gone: so is the memory
            return true;
        }
        const int fd = open_keeper_fd(pid, fdn);
        if (fd < 0) return false;
        bool removed = false;
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            struct stat st{};
            if (fstat(fd, &st) == 0) freed += (uint64_t) st.st_size;
            kill(pid, SIGTERM);
            unlink(full.c_str());
            removed = true;
        }
        close(fd);
        return removed;
    }
    const size_t tmp = entry.find(".tmp.");
    if (tmp != std::string::npos) {   // a creator's file before its rename: only when that creator is gone
        if (pid_alive((pid_t) std::atoi(entry.c_str() + tmp + 5))) return false;
    }
    const int fd = ::open(full.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    bool removed = false;
    if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
        struct stat a{}, b{};
        if (fstat(fd, &a) == 0 && lstat(full.c_str(), &b) == 0 && a.st_ino == b.st_ino && a.st_dev == b.st_dev) {
            freed += (uint64_t) a.st_size;
            removed = unlink(full.c_str()) == 0;
        }
    }
    close(fd);
    return removed;
}

// every object of `name` with another key (another pack, quant or file version): deleted when unused, so a stale
// arena gives its RAM back instead of sitting beside the new one
std::string remove_stale(const std::string& dir, const std::string& name, uint64_t key) {
    const std::string prefix = "strata-arena-" + name + "-";
    const std::string mine = prefix + hex16(key);
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return {};
    std::vector<std::string> entries;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.compare(0, prefix.size(), prefix) != 0) continue;
        if (n == mine || n == mine + ".keeper") continue;
        if (n.compare(0, mine.size(), mine) == 0 && n.find(".tmp.") == std::string::npos) continue;
        entries.push_back(n);
    }
    closedir(d);
    int removed = 0, kept = 0;
    uint64_t freed = 0;
    for (const auto& n : entries) (remove_unused(dir, n, freed) ? removed : kept)++;
    if (removed == 0 && kept == 0) return {};
    char b[256];
    std::snprintf(b, sizeof b, "; %d stale object(s) of this name deleted (%.1f GiB)%s", removed, (double) freed / kGiB,
                  kept > 0 ? (", " + std::to_string(kept) + " kept: in use by another engine").c_str() : "");
    return b;
}

}  // namespace

ArenaShm::~ArenaShm() {
    if (owns_map_ && map_ != nullptr) munmap(map_, (size_t) map_bytes_);
    map_ = nullptr;
    if (created_ && !complete_) discard_created();
    if (fd_ >= 0) close(fd_);   // (the lock goes with it)
}

void ArenaShm::discard_created() {
    if (memfd_) {
        if (keeper_ > 0) {
            kill(keeper_, SIGTERM);
            (void) waitpid(keeper_, nullptr, 0);
        }
        unlink(path_.c_str());
    } else if (!path_.empty()) {
        unlink(path_.c_str());
    }
    created_ = false;
}

bool ArenaShm::open(const std::string& name, uint64_t bytes, uint64_t key, const std::string& desc, std::string& note) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!valid_name(name)) {
        note = "STRATA_ARENA_SHM=\"" + name + "\" is not a name of [A-Za-z0-9._-] (at most 64)";
        return false;
    }
    if (key == 0 || bytes == 0) {
        note = "no key for the shared arena";
        return false;
    }
    const std::string dir = shm_dir();
    const std::string base = dir + "/strata-arena-" + name + "-" + hex16(key);
    const std::string index = base + ".keeper";
    const uint64_t file_bytes = kHdrBytes + bytes;
    std::string extra = remove_stale(dir, name, key);

    for (int attempt = 0; attempt < 4 && fd_ < 0; ++attempt) {
        // 1. attach: the file, else the keeper's memfd
        int fd = ::open(base.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        bool memfd = false;
        pid_t kp = 0;
        if (fd < 0) {
            int fdn = -1;
            if (read_keeper_index(index, kp, fdn)) {
                fd = open_keeper_fd(kp, fdn);
                memfd = fd >= 0;
                if (fd < 0) unlink(index.c_str());   // a dead keeper's index
            }
        }
        if (fd >= 0) {
            if (!lock_shared_wait(fd, memfd ? "the shared arena memfd" : base)) {
                close(fd);
                note = "the shared arena " + base + " stayed locked by another engine";
                return false;
            }
            const int state = header_state(fd, bytes, key);
            if (state == (int) kStateComplete) {
                fd_ = fd;
                memfd_ = memfd;
                keeper_ = kp;
                path_ = memfd ? index : base;
                reused_ = true;
                where_ = memfd ? "memfd held by pid " + std::to_string((int) kp) : base;
                break;
            }
            // filling but nobody holds it (its creator died), or not ours: rebuilt when unused, else a private arena
            if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                close(fd);
                note = "the shared arena " + base + " is incomplete or does not match and another engine holds it";
                return false;
            }
            if (memfd) {
                if (is_keeper(kp)) kill(kp, SIGTERM);
                unlink(index.c_str());
            } else {
                unlink(base.c_str());
            }
            close(fd);
            extra += "; an incomplete or mismatched object was deleted";
            continue;
        }

        // 2. create: a /dev/shm file when it fits (with 1 GiB to spare), else a memfd and its keeper
        struct statvfs vs{};
        const uint64_t avail = statvfs(dir.c_str(), &vs) == 0 ? (uint64_t) vs.f_bavail * (uint64_t) vs.f_frsize : 0;
        const bool use_file = avail >= file_bytes + kGiB && std::getenv("STRATA_ARENA_SHM_MEMFD") == nullptr;
        std::string err;
        if (use_file) {
            const std::string tmp = base + ".tmp." + std::to_string((int) getpid());
            fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd < 0) {
                note = "cannot create " + tmp + ": " + std::strerror(errno);
                return false;
            }
            flock(fd, LOCK_EX);
            if (ftruncate(fd, (off_t) file_bytes) != 0) err = std::string("ftruncate: ") + std::strerror(errno);
            if (err.empty()) (void) allocate_all(fd, file_bytes, err);
            Header h{};
            std::memcpy(h.magic, kMagic, sizeof kMagic);
            h.version = kVersion;
            h.state = kStateFilling;
            h.header_bytes = kHdrBytes;
            h.arena_bytes = bytes;
            h.key = key;
            h.created_unix = (uint64_t) std::time(nullptr);
            h.creator_pid = (int64_t) getpid();
            std::snprintf(h.desc, sizeof h.desc, "%s", desc.c_str());
            if (err.empty() && pwrite(fd, &h, sizeof h, 0) != (ssize_t) sizeof h) err = "header write failed";
            if (!err.empty()) {
                unlink(tmp.c_str());
                close(fd);
                note = "cannot allocate " + std::to_string(file_bytes >> 20) + " MiB in " + dir + " (" + err + ")";
                return false;
            }
            if (syscall(SYS_renameat2, AT_FDCWD, tmp.c_str(), AT_FDCWD, base.c_str(), RENAME_NOREPLACE) != 0) {
                unlink(tmp.c_str());
                close(fd);
                if (errno == EEXIST) continue;   // another engine created it meanwhile: attach to that one
                note = "cannot publish " + base + ": " + std::strerror(errno);
                return false;
            }
            fd_ = fd;
            path_ = base;
            where_ = base;
        } else {
            const std::string mname = "strata-arena-" + name + "-" + hex16(key);
            fd = (int) syscall(SYS_memfd_create, mname.c_str(), 1u /* MFD_CLOEXEC */);
            if (fd < 0) {
                note = std::string("memfd_create: ") + std::strerror(errno);
                return false;
            }
            flock(fd, LOCK_EX);
            if (ftruncate(fd, (off_t) file_bytes) != 0) err = std::string("ftruncate: ") + std::strerror(errno);
            if (err.empty()) (void) allocate_all(fd, file_bytes, err);
            Header h{};
            std::memcpy(h.magic, kMagic, sizeof kMagic);
            h.version = kVersion;
            h.state = kStateFilling;
            h.header_bytes = kHdrBytes;
            h.arena_bytes = bytes;
            h.key = key;
            h.created_unix = (uint64_t) std::time(nullptr);
            h.creator_pid = (int64_t) getpid();
            std::snprintf(h.desc, sizeof h.desc, "%s", desc.c_str());
            if (err.empty() && pwrite(fd, &h, sizeof h, 0) != (ssize_t) sizeof h) err = "header write failed";
            // the keeper gets an open file description of its own (a dup would share our flock), as fd 3, and nothing else
            char self[64];
            std::snprintf(self, sizeof self, "/proc/self/fd/%d", fd);
            int kfd = err.empty() ? ::open(self, O_RDWR | O_CLOEXEC) : -1;
            if (err.empty() && kfd < 0) err = std::string("reopen: ") + std::strerror(errno);
            if (kfd == 3) {   // dup2(3, 3) would leave FD_CLOEXEC set
                const int k2 = fcntl(kfd, F_DUPFD_CLOEXEC, 4);
                close(kfd);
                kfd = k2;
            }
            pid_t pid = 0;
            if (err.empty()) {
                posix_spawn_file_actions_t fa;
                posix_spawnattr_t at;
                posix_spawn_file_actions_init(&fa);
                posix_spawnattr_init(&at);
                posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
                posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
                posix_spawn_file_actions_adddup2(&fa, 1, 2);
                posix_spawn_file_actions_adddup2(&fa, kfd, 3);
                posix_spawn_file_actions_addclosefrom_np(&fa, 4);
                sigset_t none, def;
                sigemptyset(&none);
                sigemptyset(&def);
                sigaddset(&def, SIGTERM);
                sigaddset(&def, SIGINT);
                sigaddset(&def, SIGHUP);
                posix_spawnattr_setsigmask(&at, &none);
                posix_spawnattr_setsigdefault(&at, &def);
                // its own session: the engine's process group is ended by the harness; the keeper must outlive it
                posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
                char a0[] = "sleep", a1[] = "infinity";
                char* argv[] = {a0, a1, nullptr};
                char* envp[] = {nullptr};
                const int rc = posix_spawnp(&pid, "sleep", &fa, &at, argv, envp);
                posix_spawn_file_actions_destroy(&fa);
                posix_spawnattr_destroy(&at);
                if (rc != 0) err = std::string("cannot start the keeper: ") + std::strerror(rc);
            }
            if (kfd >= 0) close(kfd);
            if (err.empty()) {
                const std::string itmp = index + ".tmp." + std::to_string((int) getpid());
                std::FILE* f = std::fopen(itmp.c_str(), "w");
                bool ok = f != nullptr && std::fprintf(f, "%d 3\n", (int) pid) > 0;
                if (f != nullptr) ok = std::fclose(f) == 0 && ok;
                if (ok && syscall(SYS_renameat2, AT_FDCWD, itmp.c_str(), AT_FDCWD, index.c_str(), RENAME_NOREPLACE) != 0) {
                    const int e = errno;
                    unlink(itmp.c_str());
                    kill(pid, SIGTERM);
                    (void) waitpid(pid, nullptr, 0);
                    close(fd);
                    if (e == EEXIST) continue;
                    note = "cannot publish " + index + ": " + std::strerror(e);
                    return false;
                }
                if (!ok) {
                    unlink(itmp.c_str());
                    err = "cannot write " + index;
                }
            }
            if (!err.empty()) {
                if (pid > 0) {
                    kill(pid, SIGTERM);
                    (void) waitpid(pid, nullptr, 0);
                }
                close(fd);
                note = "cannot set up a memfd arena of " + std::to_string(file_bytes >> 20) + " MiB (" + err + ")";
                return false;
            }
            fd_ = fd;
            memfd_ = true;
            keeper_ = pid;
            path_ = index;
            where_ = "memfd held by pid " + std::to_string((int) pid) + " (" + index + ")";
        }
        created_ = true;
    }
    if (fd_ < 0) {
        note = "the shared arena " + base + " could not be attached or created";
        return false;
    }

    // map it 2 MiB aligned (a PMD can then map each 2 MiB folio), the arena 2 MiB past the start
    constexpr uint64_t kAlign = 2ull << 20;
    void* raw = mmap(nullptr, (size_t) (file_bytes + kAlign), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) {
        note = std::string("cannot reserve the address range: ") + std::strerror(errno);
        if (created_) discard_created();
        close(fd_);
        fd_ = -1;
        return false;
    }
    const uintptr_t start = (uintptr_t) raw, aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    const uintptr_t end = aligned + file_bytes, raw_end = start + file_bytes + kAlign;
    if (aligned > start) munmap(raw, aligned - start);
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* m = mmap((void*) aligned, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd_, 0);
    if (m == MAP_FAILED) {
        note = std::string("cannot map the shared arena: ") + std::strerror(errno);
        munmap((void*) aligned, (size_t) file_bytes);
        if (created_) discard_created();
        close(fd_);
        fd_ = -1;
        return false;
    }
    map_ = m;
    map_bytes_ = file_bytes;
    data_ = (uint8_t*) m + kHdrBytes;

    // 2 MiB folios (STRATA_ARENA_SHM_THP=0 keeps 4 KiB pages): on a new object this builds them; on a reused one the
    // folios are already huge and this only maps them by PMD (the fault path would use 4 KiB entries)
    std::string thp;
    const char* te = std::getenv("STRATA_ARENA_SHM_THP");
    if (te == nullptr || te[0] != '0') {
        const auto tc = std::chrono::steady_clock::now();
        std::atomic<uint64_t> ok_bytes{0};
        std::atomic<int> last_err{0};
        // whole 2 MiB pieces of the arena only (its end may not be 2 MiB aligned)
        const uint64_t span = bytes & ~(kAlign - 1);
        parallel_pieces(0, span, [&](uint64_t a, uint64_t len) {
            if (madvise(data_ + a, (size_t) len, MADV_COLLAPSE) == 0) ok_bytes += len;
            else last_err = errno;
        });
        char b[256];
        std::snprintf(b, sizeof b, "; 2 MiB folios: %.1f of %.1f GiB (MADV_COLLAPSE, %.1f s%s%s)", (double) ok_bytes / kGiB,
                      (double) span / kGiB, seconds_since(tc), last_err ? ", " : "",
                      last_err ? std::strerror(last_err) : "");
        thp = b;
    } else {
        thp = "; 4 KiB pages (STRATA_ARENA_SHM_THP=0)";
    }
    char b[256];
    std::snprintf(b, sizeof b, "%s %s in %.1f s", reused_ ? "attached to" : "created", where_.c_str(), seconds_since(t0));
    note = b + thp + extra;
    return true;
}

bool ArenaShm::mark_complete(std::string& note) {
    if (fd_ < 0 || !created_) return false;
    const uint32_t s = kStateComplete;
    if (pwrite(fd_, &s, sizeof s, (off_t) offsetof(Header, state)) != (ssize_t) sizeof s) {
        note = std::string("cannot mark the shared arena complete: ") + std::strerror(errno);
        return false;
    }
    complete_ = true;
    flock(fd_, LOCK_SH);   // the next engines may attach now
    return true;
}

bool ArenaShm::drop_if_unused() {
    if (fd_ < 0) return false;
    if (flock(fd_, LOCK_EX | LOCK_NB) != 0) return false;
    if (memfd_) {
        if (is_keeper(keeper_)) kill(keeper_, SIGTERM);
        unlink(path_.c_str());
    } else {
        unlink(path_.c_str());
    }
    return true;
}

#endif

}  // namespace strata::core

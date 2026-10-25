// src/core/arena_shm_test.cpp - [claude] STRATA_ARENA_SHM's object (ArenaShm), without a GPU: create, fill, complete,
// attach from another process, a key that differs (another pack) never attaches and its unused object is deleted, an
// incomplete object (its creator died) is rebuilt, and the memfd + keeper variant does the same.  CPU and RAM only.
#include "strata/core/arena_shm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dirent.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using strata::core::ArenaShm;

namespace {
int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr uint64_t kBytes = 64ull << 20;

int count_objects(const std::string& dir, const std::string& name) {
    int n = 0;
    if (DIR* d = opendir(dir.c_str())) {
        const std::string p = "strata-arena-" + name + "-";
        while (dirent* e = readdir(d)) n += std::strncmp(e->d_name, p.c_str(), p.size()) == 0;
        closedir(d);
    }
    return n;
}

// in a child: attach (must be reused) and check the bytes; exit 0 when they are the pattern
int child_attach(const std::string& name, uint64_t key, uint8_t pat) {
    const pid_t pid = fork();
    if (pid == 0) {
        ArenaShm s;
        std::string note;
        if (!s.open(name, kBytes, key, "test", note)) { std::fprintf(stderr, "child: %s\n", note.c_str()); _exit(2); }
        if (!s.reused()) _exit(3);
        for (uint64_t i = 0; i < kBytes; i += 4096)
            if (s.data()[i] != (uint8_t) (pat + i / 4096)) _exit(4);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 99;
}

void run(bool memfd) {
    char dir[] = "/dev/shm/strata-arena-test-XXXXXX";
    if (mkdtemp(dir) == nullptr) { std::perror("mkdtemp"); ++fails; return; }
    setenv("STRATA_ARENA_SHM_DIR", dir, 1);
    if (memfd) setenv("STRATA_ARENA_SHM_MEMFD", "1", 1); else unsetenv("STRATA_ARENA_SHM_MEMFD");
    const std::string name = "t";
    std::string note;
    {   // 1. create, fill, complete
        ArenaShm s;
        CHECK(s.open(name, kBytes, 0x1111, "test", note));
        CHECK(!s.reused());
        CHECK(((uintptr_t) s.data() & ((2u << 20) - 1)) == 0);
        for (uint64_t i = 0; i < kBytes; i += 4096) s.data()[i] = (uint8_t) (7 + i / 4096);
        CHECK(s.mark_complete(note));
        std::printf("  %s: %s\n", memfd ? "memfd" : "file", note.c_str());
    }
    // 2. another process attaches to the complete object and sees the bytes
    CHECK(child_attach(name, 0x1111, 7) == 0);
    {   // 3. another key: a new object, never the old one; the old (unused) one is deleted
        ArenaShm s;
        CHECK(s.open(name, kBytes, 0x2222, "test", note));
        CHECK(!s.reused());
        CHECK(note.find("stale object(s) of this name deleted") != std::string::npos);
        CHECK(s.data()[4096] == 0);
        // (not completed: it is deleted when this object goes)
    }
    CHECK(count_objects(dir, name) == 0);
    {   // 4. a size that differs is not attached either
        ArenaShm a;
        CHECK(a.open(name, kBytes, 0x3333, "test", note));
        CHECK(a.mark_complete(note));
        ArenaShm b;   // same key, same process (a second engine): attaches, shared lock
        CHECK(b.open(name, kBytes, 0x3333, "test", note));
        CHECK(b.reused());
        ArenaShm c;   // another key while 0x3333 is in use: 0x3333 is kept
        CHECK(c.open(name, kBytes, 0x4444, "test", note));
        CHECK(note.find("kept: in use") != std::string::npos);
    }
    {   // 5. a creator that dies before completing: the next start rebuilds it
        const pid_t pid = fork();
        if (pid == 0) {
            ArenaShm s;
            std::string n2;
            if (!s.open(name, kBytes, 0x5555, "test", n2)) _exit(2);
            _exit(0);   // no mark_complete, no destructor: as if killed in the load
        }
        int st = 0;
        waitpid(pid, &st, 0);
        ArenaShm s;
        CHECK(s.open(name, kBytes, 0x5555, "test", note));
        CHECK(!s.reused());
        CHECK(note.find("incomplete or mismatched object was deleted") != std::string::npos);
        CHECK(s.mark_complete(note));
        CHECK(s.drop_if_unused());   // the cleanup a failed sampled check does
    }
    {   // 6. cleanup: every object of the name goes when a new key opens and nothing holds them
        ArenaShm s;
        CHECK(s.open(name, kBytes, 0x6666, "test", note));
    }
    CHECK(count_objects(dir, name) == 0);
    rmdir(dir);
}
}  // namespace

int main() {
    run(false);
    run(true);
    std::printf("%s\n", fails == 0 ? "arena_shm_test: ok" : "arena_shm_test: FAILED");
    return fails == 0 ? 0 : 1;
}

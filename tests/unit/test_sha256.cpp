#include <catch2/catch_test_macros.hpp>
#include <string>

#include "common/sha256.hpp"

static std::string h(const std::string& s) { return blz::to_hex(blz::sha256(s.data(), s.size())); }

TEST_CASE("sha256 FIPS 180-4 vectors") {
  CHECK(h("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(h("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // 56 bytes: the padding spills into a second block.
  CHECK(h("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(h(std::string(1'000'000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

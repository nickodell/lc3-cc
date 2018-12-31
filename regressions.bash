# set -x

run() {
  test_name=$1
  ./compile.py tests/$test_name.c > tests/expect_asm/$test_name
  echo "########################################## ASM"
  cat tests/expect_asm/$test_name
  echo "########################################## OUT"
  cat tests/expect_asm/$test_name | ./lc3run | tee tests/expect_out/$test_name
  diff tests/expect_asm/$test_name tests/expect_asm/$test_name.correct
  diff tests/expect_out/$test_name tests/expect_out/$test_name.correct
}


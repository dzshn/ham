import traceback

import ham


def main() -> None:
    while True:
        try:
            src = input("% ")
        except EOFError:
            print()
            break

        try:
            stack = ham.evaluate(src)
        except Exception:
            traceback.print_exc()
        else:
            print(ham.pretty_stack(stack))


if __name__ == "__main__":
    main()

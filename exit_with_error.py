from colorama import Fore


def exit_with_error(err: str):
    print(Fore.RED, err, Fore.RESET)
    exit(0)

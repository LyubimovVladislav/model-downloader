from colorama import Fore


def get_user_decision() -> bool:
    print('Override the existing file?')
    while True:
        user_decision = input(f'Type {Fore.RED}y/yes{Fore.RESET} to re-download and override the checkpoint file, '
                              f'{Fore.RED}n/no{Fore.RESET} to continue with existing checkpoint or '
                              f'{Fore.RED}a/abort{Fore.RESET} to exit the program.')
        user_decision = user_decision.lower().strip()
        if user_decision == 'y' or user_decision == 'yes':
            return True
        if user_decision == 'n' or user_decision == 'no':
            return False
        if user_decision == 'a' or user_decision == 'abort':
            exit(0)

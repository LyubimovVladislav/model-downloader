from typing import Optional

from colorama import Fore


def get_user_decision() -> bool:
    print('Override the existing file?')
    while True:
        user_decision = input(f'Type {Fore.RED}y/yes{Fore.RESET} to re-download and override the checkpoint file, '
                              f'{Fore.RED}n/no{Fore.RESET} to continue with existing checkpoint or '
                              f'{Fore.RED}a/abort{Fore.RESET} to exit the program.\n'
                              f'> ')
        user_decision = user_decision.lower().strip()
        if user_decision == 'y' or user_decision == 'yes':
            return True
        if user_decision == 'n' or user_decision == 'no':
            return False
        if user_decision == 'a' or user_decision == 'abort':
            exit(0)


def ask_for_base_model_link() -> Optional[str]:
    print('Cant find a LoRA base model. Please provide the base model link from Civitai or '
          f'type {Fore.RED}a/abort{Fore.RESET} to exit the program.')
    user_decision = input('> ')
    user_decision = user_decision.strip()
    if user_decision.lower() == 'a' or user_decision.lower() == 'abort':
        exit(0)
    return user_decision


def ask_what_version_to_download(name, versions) -> int:
    while True:
        print(f'Found multiple versions of the {name}.\n'
              f'Specify which version do you want to download or type {Fore.RED}a/abort{Fore.RESET}'
              f' to exit the program.\n'
              f'Options: {", ".join([f"{Fore.RED}{val + 1}{Fore.RESET} - {key}" for key, val in versions.items()])}')
        user_decision = input('> ')
        user_decision = user_decision.strip()
        if user_decision.lower() == 'a' or user_decision.lower() == 'abort':
            exit(0)
        if user_decision.isdigit() and int(user_decision) in versions.values():
            return int(user_decision) - 1

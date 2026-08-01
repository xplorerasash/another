"""Small CLI demo to interact with the moderation pipeline in real-time.
"""
from pathlib import Path

def main():
    from chatbot import process_message

    USER_ID = 'cli_user'
    print('SafeChat-AI CLI demo. Type quit to exit.')
    while True:
        try:
            msg = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nGoodbye!')
            break
        if msg.lower() in {'quit', 'exit'}:
            print('Goodbye!')
            break
        res = process_message(USER_ID, msg)
        print('Bot:', res['reply'])

if __name__ == '__main__':
    main()

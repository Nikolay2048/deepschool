

def read_file_test():
    with open("war_and_peace.ru.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print(text[:500])

read_file_test()
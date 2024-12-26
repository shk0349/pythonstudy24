import random

def main(count):
    print("=" * 20)
    while True:
        print("=" * 20)
        s_num = input("1. 수동\n2.자동\n")

        if s_num == '1':
            print("=" * 20)
            for i in range(count):
                lotto_num = manual()
                print("로또번호 : ",lotto_num)
            break

        elif s_num == '2':
            print("=" * 20)
            for i in range(count):
                lotto_num = auto()
                print("로또번호 : ", lotto_num)
            break

        else:
            print("=" * 20)
            print("다시 입력해주세요.")
            print("=" * 20, end="\n")

        print("=" * 20, end="\n")

def manual():
    while True:
        num1 = int(input("첫번째 수"))
        num2 = int(input("두번째 수"))
        num3 = int(input("세번째 수"))
        num4 = int(input("네번째 수"))
        num5 = int(input("다섯번째 수"))
        num6 = int(input("여섯번째 수"))
        print("=" * 20, end="\n")

        manual = [num1, num2, num3, num4, num5, num6]
        dupli = check(manual)

        if dupli == False:
            print("\n중복된 값이 있습니다. 다시 입력해주세요.\n")
            print("=" * 20, end="\n")
            continue

        else:
            manual.sort()

        if manual[-1] > 45:
            print("\n45이하의 숫자로 다시 입력해주세요.")
            print("=" * 20, end="\n")
            continue

        else:
            break

    return manual

def auto():
    while True:
        num1 = random.randrange(1, 45)
        num2 = random.randrange(1, 45)
        num3 = random.randrange(1, 45)
        num4 = random.randrange(1, 45)
        num5 = random.randrange(1, 45)
        num6 = random.randrange(1, 45)

        auto = [num1, num2, num3, num4, num5, num6]
        dupli = check(auto)

        if dupli == False:
            continue

        else:
            auto.sort()
            break
    return auto

def check(lotto_num):
    i = 0
    dupli = True

    while i < 45:
        if lotto_num.count(i) < 2:
            i += 1
            continue
        else:
            dupli = False
            break

    return dupli
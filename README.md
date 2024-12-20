# pythonstudy24
Python AI 기초학습용

MBC 아카데미 컴퓨터 교육센터 수원점에서 AI기초 학습으로 Python 학습 진행용

https://wikidocs.net/book/1


"""
        americano = 30
        latte = 15
        greentea = 25
        americano_price = 2000
        latte_price = 2500
        greentea_price = 3000
        
        def print_auth():
            auth = input("""
            1. 손님
            2. 관리자
            3. 프로그램 종료
            """)
            return auth
        
        def print_guest_order():
            guest_order = input("""
            1. 메뉴보기
            2. 주문하기
            """)
            return guest_order
        
        def print_menu():
            print("1. 아메리카노 : %d원" % americano_price)
            print("2. 라떼 : %d원" % latte_price)
            print("3. 녹차 : %d원" % greentea_price)
        
        def print_order(menu):
            global americano, latte, greentea
            if menu == '1':
                if americano > 0:
                    americano -= 1
                    print(f"주문하신 아메리카노 1잔 나왔습니다. 남은 수량: {americano}잔")
                else:
                    print("아메리카노가 모두 소진되었습니다.")
            elif menu == '2':
                if latte > 0:
                    latte -= 1
                    print(f"주문하신 라떼 1잔 나왔습니다. 남은 수량: {latte}잔")
                else:
                    print("라떼가 모두 소진되었습니다.")
            elif menu == '3':
                if greentea > 0:
                    greentea -= 1
                    print(f"주문하신 녹차 1잔 나왔습니다. 남은 수량: {greentea}잔")
                else:
                    print("녹차가 모두 소진되었습니다.")
            else:
                print("잘못된 메뉴입니다.")
        
        def print_admin():
            global americano, latte, greentea
            print(f"""
            판매량
            1. 아메리카노 : {30 - americano}잔
            2. 라떼 : {15 - latte}잔
            3. 녹차 : {25 - greentea}잔
            """)
            total = (americano_price * (30 - americano)) + (latte_price * (15 - latte)) + (greentea_price * (25 - greentea))
            print(f"총 매출액 : {total}원")
        
        def main():
            print("매장 방문을 환영합니다.")
            auth = print_auth()
            while True:
                
                if auth == '1':
                    guest_order = print_guest_order()
                    if guest_order == '1':
                        print_menu()
                    elif guest_order == '2':
                        while True:
                            print_menu()
                            menu = input("주문할 메뉴를 선택해주세요 (1~3): ")
                            print_order(menu)
                            extra = input("추가 주문하시겠습니까? (Y/N): ").lower()
                            if extra == 'n':
                                break
                    else:
                        print("잘못된 입력입니다.")
                elif auth == '2':
                    print_admin()
                elif auth == '3':
                    print("프로그램을 종료합니다.")
                    break
                else:
                    print("잘못된 입력입니다.")
        
        main()
"""

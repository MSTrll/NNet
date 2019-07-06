def fc(a, b):
    print("fc", a, b)

class A:
    def __init__(self, func=fc):
        self.digit = 0
        self.func = func

    def caller(self):
        self.func(1, 2)

CA = A()
CA.caller()
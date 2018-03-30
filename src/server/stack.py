import sys

class Empty(Exception):
    pass

class Stack:
    def __init__(self, size=None):
        self.__element_list = list()

        if size:
            self.size = size
        else:
            self.size = sys.maxsize

    def peek(self):
        if len(self.__element_list) == 0:
            raise Empty

        return self.__element_list[len(self.__element_list) - 1]

    def push(self, element):
        if len(self.__element_list) >= self.size:
            del self.__element_list[0]
        self.__element_list.append(element)

    def pop(self):
        if len(self.__element_list) == 0:
            raise Empty

        element = self.peek()

        del self.__element_list[len(self.__element_list) - 1]

        return element

    def yield_generator(self):
        for i in range(len(self.__element_list) - 1, -1, -1):
            yield self.__element_list[i]

class Accumulator():

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0

    def add(self, val, cnt):
        self.value += val
        self.count += cnt

    def get_average(self):
        if self.count == 0:
            return -1
        return self.value / self.count

    def get_accumulated_value(self):
        return self.value

    def get_accumulated_count(self):
        return self.count

    def get_name(self):
        return self.name

    def __str__(self):
        return "[{}] value: {}, count: {}".format(self.name, self.value, self.count)

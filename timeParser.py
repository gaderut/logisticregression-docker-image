class timeParser:

    def __init__(self, timestring):
        sp = timestring.split(':')
        if len(sp) == 2:
            self.hour = float(sp[0])
            self.min = float(sp[1])
        else:
            self.hour = float(timestring)
            self.min = 0.0

    def __lt__(self, other):

        return self.hour + self.min < other.hour + other.min

    def __le__(self, other):

        return self.hour + self.min <= other.hour + other.min

    def __gt__(self, other):

        return self.hour + self.min > other.hour + other.min

    def __ge__(self, other):

        return self.hour + self.min >= other.hour + other.min

    def __eq__(self, other):

        return self.hour + self.min == other.hour + other.min

    def __ne__(self, other):

        return self.hour + self.min != other.hour + other.min

    def __add__(self, other):

        return timeParser(str(int(self.hour + other.hour)) + ':' + str(int(self.min + other.min)))

    def __sub__(self, other):

        return timeParser(str(int(self.hour - other.hour)) + ':' + str(int(self.min - other.min)))
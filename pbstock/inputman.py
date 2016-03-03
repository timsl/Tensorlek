#! /usr/bin/python
import csv

class Inputman:
    def __init__(self, n_input, n_classes):
        self.n_input = n_input
        self.n_classes = n_classes
        self.index = 0
        #load input data
        input_data = list()
        with open('smp.csv', 'rb') as csvfile:
            closereader = csv.reader(csvfile)
            for row in closereader:
                try:
                    input_data.append(float(row[4]))
                except:
                    print("error parsing, \"", row[4], "\".")
        input_data.reverse()
        self.data_max = len(input_data)
        self.input_data= input_data 


    def reset(self):
        self.index = 0
    
    def set(self, index):
        self.index = index

    def next_pair(self):
        x = [0] * self.n_input
        y = [0] * self.n_classes
        ptr = self.index
        xlen = self.n_input
        ylen = self.n_classes

        if (self.index + self.n_classes + self.n_input) > self.data_max:
            return [], 0

        for i in range(self.n_input):
            x[i] = self.input_data[ptr+i]
        ptr += self.n_input
        for i in range(self.n_classes):
            y[i] = self.input_data[ptr+i]
        
        self.index += 1  # 123(4) 234(5) 345(6)
        #ptr += self.n_classes
        #self.index = ptr # 123(4) 567(8)
        
        return [x], [y]

    def next_norm(self):
        x = [0] * self.n_input
        y = [0] * self.n_classes
        ptr = self.index
        xlen = self.n_input
        ylen = self.n_classes

        if (self.index + self.n_classes + self.n_input) > self.data_max:
            return [], 0

        for i in range(self.n_input):
            x[i] = (self.input_data[ptr+i] - self.min_price) / (self.max_price - self.min_price)
        ptr += self.n_input
        for i in range(self.n_classes):
            y[i] = (self.input_data[ptr+i] - self.min_price) / (self.max_price - self.min_price)
        
        self.index += 1  # 123(4) 234(5) 345(6)
        #ptr += self.n_classes
        #self.index = ptr # 123(4) 567(8)
        
        return [x], [y]
    
    def set_interval(self, _min, _max):
        self.min_price = _min
        self.max_price = _max

    def set_norm_range(self, _min=-1, _max=1):
        self.nmin = _min
        self.nmax = _max

    def denorm(self, val):
        for i in range(0, len(val)):
            for j in range(0, len(val[i])):
                ymin, ymax = self.nmin, self.nmax
                xmin, xmax = self.min_price, self.max_price
                x = val[i][j]
                val[i][j] = ((x-ymin)/(ymax-ymin))*(xmax-xmin)+xmin
        return val
    
    def norm(self, val):
        for i in range(0, len(val)):
            for j in range(0, len(val[i])):
                ymin, ymax = self.nmin, self.nmax
                xmin, xmax = self.min_price, self.max_price
                x = val[i][j]
                val[i][j] = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
        return val

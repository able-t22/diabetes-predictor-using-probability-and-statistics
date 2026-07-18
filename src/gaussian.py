import math

def gaussian_probability(x, mean, std):

    exponent = math.exp(-((x-mean)**2)/(2*(std**2)))

    return (1/(math.sqrt(2*math.pi)*std))*exponent
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from function_layer import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax= 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_fruit_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
fruit_price = add_fruit_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(fruit_price, tax)

dprice = 1
dall_price , dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_fruit_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple_num, dapple, dorange, dorange_num, dtax)

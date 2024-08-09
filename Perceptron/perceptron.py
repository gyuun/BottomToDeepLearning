"""describing perceptron node"""
import numpy as np

class PerceptronLogic :
    """논리게이트를 퍼셉트론으로 나타낸 클래스"""
    def __init__ (self, x):
        """생성자"""
        # x is nd.array()
        self.x = np.append(x, 1)

    def set_attribute(self, x):
        """입력값 변경"""
        self.x = np.append(x, 1)

    def result(self, w):
        """신경곱 결과"""
        if np.sum(np.dot(self.x, w) > 0) :
            return 1
        return 0

    def and_gate(self):
        """AND 게이트"""
        w = np.array([0.5, 0.5, -0.7])

        return self.result(w)

    def nand_gate(self):
        """NAND 게이트"""
        w = np.array([-0.5, -0.5, 0.7])
        return self.result(w)

    def or_gate(self):
        """OR 게이트"""
        w = np.array([0.5, 0.5, -0.3])

        return self.result(w)

    def xor_gate(self):
        """XOR 게이트"""
        s1 = self.nand_gate()
        s2 = self.or_gate()
        new_neuron = PerceptronLogic(np.array([s1, s2]))

        return new_neuron.and_gate()

  


"""Testing network"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Network import network

def test_stair(x):
    """계단함수 테스트"""
    return network.stair(x)


def test_ReLu(x):
    """렐루함수 테스트"""
    return network.relu(x)

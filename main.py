#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/05/02 20:48:00
# @Author  : Hz6826
# @File    : main.py
# @Software: PyCharm

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 定义行星类
class Planet:
    def __init__(self, planet_name, planet_mass, initial_position=(0, 0, 0), initial_velocity=(0, 0, 0)):
        self.name = planet_name
        self.loc = np.array(initial_position)  # 使用NumPy数组来存储位置
        self.v = np.array(initial_velocity)  # 使用NumPy数组来存储速度
        self.a = np.array([0, 0, 0])  # 加速度
        self.force = np.array([0, 0, 0])  # 作用力
        self.m = planet_mass  # 行星的质量
        self.trail = []  # 存储轨迹点的列表

    def update_location(self, dt, all_planets):
        for planet in all_planets:
            self.force = np.array([0, 0])  # 重置作用力
            # 排除自身
            if planet is self:
                continue
            # 计算两行星之间的距离
            r_val = np.linalg.norm(np.array(self.loc) - np.array(planet.loc))
            # 计算引力
            self.force += G * self.m * planet.m / r_val ** (p) * (planet.loc - self.loc) / r_val

        # 更新加速度
        self.a = self.force / self.m
        # 更新速度
        self.v += self.a * dt
        # 更新位置
        self.loc += self.v * dt
        # 将当前位置添加到轨迹列表中
        self.trail.append(self.loc.tolist())


# 读取metadata.json文件
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

G = 6.67430e-11  # 引力常数
p = 2  # 引力定律
dt = 3600 * 24  # 时间步长 (1天)
steps = 1000  # 时间步数
total_time = steps * dt  # 总时间
# 读取初始时间，格式为2022-01-01T00:00:00Z，并转化为Unix时间戳
initial_time = pd.to_datetime(metadata['initial_time']).timestamp()


# 创建行星对象
planets = []
for planet_data in metadata['planets']:
    name = planet_data['name']
    mass = planet_data['mass']
    position = planet_data['position']
    velocity = planet_data['velocity']
    planet = Planet(name, mass, position, velocity)
    planets.append(planet)


time_array = np.arange(initial_time, initial_time + total_time, dt)
time_df = pd.DataFrame(time_array, columns=['time'])
# 开始模拟
for i in range(steps):
    for planet in planets:
        planet.update_location(dt, planets)


# 绘制轨迹
for planet in planets:
    x = [p[0] for p in planet.trail]
    y = [p[1] for p in planet.trail]
    plt.plot(x, y, label=planet.name)

plt.legend()
plt.show()

# 保存轨迹数据
for planet in planets:
    df = pd.DataFrame(planet.trail, columns=['x', 'y', 'z'])
    df_merged = pd.merge(df, time_df, on='key', how='outer') # 合并时间戳
    df_merged.to_csv(f'{planet.name}_trail.csv', index=False)

print('模拟完成！')



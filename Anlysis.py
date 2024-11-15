import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

episodes = []
rewards = []

# 从文件中读取数据
with open('episode_reward_8.txt', 'r') as f:
    for line in f:
        # 提取奖励
        reward_part = line.split('reward is')[-1].split(' ')[1].strip()  # 获取奖励值
        rewards.append(int(reward_part))

        # 提取回合
        episode_part = line.split('epsidode')[-1].strip()  # 获取回合值
        episodes.append(int(episode_part))

# 打印提取的回合和奖励
print("Episodes:", episodes)
print("Rewards:", rewards)

# 绘制图表
plt.plot(episodes, rewards, label='Rewards')
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
# plt.savefig('total_reward_plot_6.png')  # 保存图表为 PNG 文件
plt.show()

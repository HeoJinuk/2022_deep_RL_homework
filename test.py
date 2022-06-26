import argparse
import os
import gym
import time
import random
import numpy as np
from model import A3CTestAgent
from utils import pre_processing

# Training Parameters
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env-name', default='ALE/Pong-v5', metavar='ENV',
                    help='environment to train on (default: ALE/Pong-v5)')
parser.add_argument('--model-path', default='./save_model_ALE/Pong-v5/model', metavar='MODEL',
                    help='model load (default: ./save_model_ALE/Pong-v5/model)')

if __name__ == "__main__":
    args = parser.parse_args()

    # 테스트를 위한 환경, 모델 생성
    env = gym.make(args.env_name, render_mode='human')
    state_size = (84, 84, 4)
    action_size = env.action_space.n
    model_path = args.model_path
    render = True

    agent = A3CTestAgent(action_size, state_size, model_path)

    num_episode = 10
    for e in range(num_episode):
        done = False
        dead = False

        score, start_life = 0, 5
        observe = env.reset()

        # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)

        # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
        state = pre_processing(observe)
        history = np.stack([state, state, state, state], axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if render:
                env.render(mode='rgb_array')
                time.sleep(0.05)

            # 정책 확률에 따라 행동을 선택
            action, policy = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            # 죽었을 때 시작하기 위해 발사 행동을 함
            if dead:
                action, dead = 1, False

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(action)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['lives']:
                dead, start_life = True, info['lives']

            score += reward

            if dead:
                history = np.stack((next_state, next_state,
                                    next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                # 각 에피소드 당 학습 정보를 기록
                print("episode: {:3d} | score : {:4.1f}".format(e, score))

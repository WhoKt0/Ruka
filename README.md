# Лабораторная работа №6: Visual Reinforcement Learning для управления роботом

Этот проект реализует среду Gymnasium для обучения манипулятора по сырым пикселям
в симуляторе PyBullet, а также скрипт обучения на базе Stable-Baselines3.

## Основные компоненты
- **envs/robot_arm_env.py** — среда `RobotArmEnv` c камерой, frame stacking и dense/sparse вознаграждением.
- **models/vision.py** — модуль с переносом обучения (MobileNetV3-Small) для ускорения сходимости.
- **train.py** — точка входа для обучения PPO с выбором бэкбона зрения.

## Требования
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск обучения
Минимальный пример с лёгкой CNN (NatureCNN) и серыми кадрами 84×84:
```bash
python train.py --vision custom --total-timesteps 50000 --frame-stack 4 --frame-skip 4 --grayscale
```

Для transfer learning с MobileNet (быстрый старт, но требовательнее к ресурсам GPU):
```bash
python train.py --vision mobilenet --total-timesteps 20000 --frame-stack 3 --image-size 84 --grayscale
```

Полезные флаги:
- `--eye-in-hand` — камера на схвате, иначе вид сверху.
- `--gui` — визуализация PyBullet GUI (медленнее).
- `--num-envs` — количество параллельных окружений (при достаточных ресурсах).

## Наблюдения и действия
- Наблюдение: стек из N последних кадров (серых 1-канальных) + углы 7 приводов.
- Действие: непрерывный вектор смещения `(Δx, Δy, Δz)` для конечного звена.

## Вознаграждение
`reward = -w1 * distance + contact_bonus (при касании) - time_penalty`

## Рекомендации по экспериментам
- Разрешение 64×64 или 84×84 для ускорения симуляции.
- Frame skip 4–8 и frame stack 3–4 для учёта динамики.
- Начните с dense reward (штраф за расстояние), затем усиливайте sparse награду за контакт.

## Отчёт
В отчёте укажите конфигурацию камеры (над столом или eye-in-hand), выбранную архитектуру
(Custom CNN или MobileNet), графики TensorBoard и наблюдения по скорости сходимости.

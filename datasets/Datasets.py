import pandas as pd
import random

random.seed(42)

industries = ['Искусственный интеллект', 'Медицина', 'Сельское хозяйство', 'Финтех', 'Возобновляемая энергия']
stages = ['Идея', 'Сид', 'Рост', 'Масштабирование']
locations = ['США', 'Европа', 'Азия', 'Африка', 'Южная Америка']
eligibility = ['Опыт > 2 лет', 'Только США', 'Сид стадия', 'Выручка > $1M']

grants = [
    {
        'ID гранта': f'G-{i+1}',
        'Сумма гранта': random.randint(10000, 1000000),
        'Сфера интересов': random.choice(industries),
        'Критерии': random.choice(eligibility),
        'Дедлайн': f"2024-{random.randint(1, 12):02}-{random.randint(1, 28):02}",
        'Тип': random.choice(['Сид', 'Рост', 'R&D', 'Социальный эффект']),
    }
    for i in range(50)
]

startups = [
    {
        'ID стартапа': f'S-{i+1}',
        'Стадия': random.choice(stages),
        'Индустрия': random.choice(industries),
        'Выручка': random.randint(0, 10000000),
        'Необходимое финансирование': random.randint(10000, 1000000),
        'Локация': random.choice(locations),
        'Опыт работы': random.randint(0, 10),
        'Размер команды': random.randint(1, 200),
        'Инновационный фокус': random.choice(['На базе ИИ', 'Устойчивое развитие', 'На блокчейне', 'С IoT']),
    }
    for i in range(100)
]

matches = []
for grant in grants:
    for startup in startups:
        if (grant['Сфера интересов'] == startup['Индустрия'] and
            grant['Сумма гранта'] >= startup['Необходимое финансирование']):
            matches.append({'ID гранта': grant['ID гранта'], 'ID стартапа': startup['ID стартапа']})

grants_df = pd.DataFrame(grants)
startups_df = pd.DataFrame(startups)
matches_df = pd.DataFrame(matches)

grants_df.to_csv('grants.csv', index=False)
startups_df.to_csv('startups.csv', index=False)
matches_df.to_csv('matches.csv', index=False)

import pandas as pd

class GrantStartupMatcher:
    def __init__(self):
        pass

    def calculate_score(self, grant, startup):
        score = 0
        if grant['Сфера интересов'] == startup['Индустрия']:
            score += 3
        if grant['Сумма гранта'] >= startup['Необходимое финансирование']:
            score += 2
        if 'Опыт > 2 лет' in grant['Критерии'] and startup['Опыт работы'] > 2:
            score += 1
        if 'Только США' in grant['Критерии'] and startup['Локация'] == 'США':
            score += 1
        return score

    def match(self, grants, startups, threshold=5):
        matches = []
        for _, grant in grants.iterrows():
            for _, startup in startups.iterrows():
                score = self.calculate_score(grant, startup)
                if score >= threshold:
                    matches.append({
                        'ID гранта': grant['ID гранта'],
                        'ID стартапа': startup['ID стартапа'],
                        'Счет': score
                    })
        return pd.DataFrame(matches)

grants_df = pd.read_csv('grants.csv')
startups_df = pd.read_csv('startups.csv')

matcher = GrantStartupMatcher()

matches_df = matcher.match(grants_df, startups_df)

matches_df.to_csv('matches_with_scores.csv', index=False)

print("Матчи сохранены в файл matches_with_scores.csv")

import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print("Witaj, ten skrypt pobiera dane z bazy zawodników FIFA 21 i tworzy model przyznawania poziomu umiejętności.")


link=input("Wprowadź ściezkę pliku fifa.xlsx: ")

print("Wczytuję dane...")

df=pd.read_excel(r(link))

poz=input("Wybierz pozycję do analizy spośród: ST,LW,RW,CAM,LM,CM,RM,CDM,LB,CB,RB,GK: ")

df=df[df["team_position"]==poz]

y=df.overall

plr_features= ['weak_foot', 'skill_moves', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

x=df[plr_features].dropna(axis=1)

overall_model = DecisionTreeRegressor()

overall_model.fit(x,y)

print("Predykcja modelu podstawowego:",overall_model.predict(x.head()))
print("Wartości podstawowe:",y.head().tolist())
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)

predicted_overall = overall_model.predict(x)
print("Błąd modelu podstawowego:",mean_absolute_error(y, predicted_overall))

print("Błąd jest równy 0, teoretycznie model idealny...")
print("Przeczodzę do walidacji modelu")

overall_model = DecisionTreeRegressor(random_state=1)
overall_model.fit(train_X, train_y)

val_predictions = overall_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Błąd modelu walidacyjnego: {:,.1f}".format(val_mae))


print("Wartości przewdywane modelu walidacyjnego: ",val_predictions[:5])

print("Wartości odniesienia modelu walidacyjnego: ")

print(val_y[:5])


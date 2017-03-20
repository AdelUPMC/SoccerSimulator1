from soccersimulator import   SoccerTeam
from Mon_projet import Fonceur, Attaquant, Defenseur

## Creation d'une equipe
team1 = SoccerTeam(name="team1")
team1.add("El Matador",Fonceur()) #Strategie Fonceur

team2 = SoccerTeam(name="team2")
team2.add("El Matador",Attaquant()) #Strategie Attaquant
team2.add("O Monstro",Defenseur()) #Strategie Defenseur


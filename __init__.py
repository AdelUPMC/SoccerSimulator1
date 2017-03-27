from soccersimulator import   SoccerTeam, Strategy
import MesStrategies

def get_team(i):
    team = SoccerTeam(name="TheLegend27")
    if (i==1):
        team.add("TheLegend27",MesStrategies.Solo()) 
    if (i==2):
        team.add("TheLegend27",MesStrategies.Attaquant()) 
        team.add("O Monstro",MesStrategies.Defenseur()) 
    if (i==4):
        team.add("El Matador",MesStrategies.Defenseur()) 
        team.add("O Monstro",MesStrategies.Solo()) 
        team.add("LeMur",MesStrategies.MVP_Milieu())
        team.add("El Matador",MesStrategies.Attaquant())
    return team 
            

## Creation d'une equipe
get_team(1)
get_team(2)
get_team(4)

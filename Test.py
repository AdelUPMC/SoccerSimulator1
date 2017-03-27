from soccersimulator.strategies  import Strategy
from soccersimulator.mdpsoccer import SoccerTeam, Simulation
from soccersimulator.gui import show_simu
from tools import Item, Action
from soccersimulator.settings import GAME_HEIGHT, GAME_WIDTH
import MesStrategies



            
## Strategie de test

class Testcoeq(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Testcoeq")
    def compute_strategy (self, state, id_team, id_player):
        tools = Action(state,id_team,id_player)
        return tools.aller_vect+tools.passe

class DefTest(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Defenseur")
    def compute_strategy (self, state, id_team, id_player):
        tools = Action(state,id_team,id_player)
        if tools.can_shoot :
            return tools.shoot_but
        if id_team == 1:
            if state.ball.position.x<(GAME_WIDTH/4.0)+10 :
                return tools.aller_vect()
        
        if state.ball.position.x>(GAME_WIDTH*(3.0/4)) :
            return tools.aller_vect()
        return tools.aller(tools.position_defenseur)
    
class Immobile(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Immobile")
    def compute_strategy (self, state, id_team, id_player):
        tools=Action(state,id_team,id_player)
        return tools.immobile
        

                 
        
    


if __name__ == "__main__":
    team1 = SoccerTeam(name="TheLegend27")
    team2 = SoccerTeam(name="Top1")
    #team1.add("Puyol",MesStrategies.Defenseur()) 
    team1.add("Zizou",MesStrategies.Solo())
    team1.add("Messi",MesStrategies.MVP_Milieu())
    #team1.add("Giuly",MesStrategies.Attaquant())
    
    
    team2.add("Atk1",MesStrategies.Attaquant())
    team2.add("Def1",MesStrategies.Defenseur())

    
    #Creation d'une partie
    simu = Simulation(team1, team2)
    #Jouer et afficher la partie
    show_simu(simu)


from soccersimulator.strategies  import Strategy
from soccersimulator.mdpsoccer import SoccerTeam, Simulation, SoccerAction
from soccersimulator.gui import show_state,show_simu
from soccersimulator.utils import Vector2D
from soccersimulator.settings import GAME_HEIGHT, GAME_WIDTH
from soccersimulator.tools import MyState 


class TestStrategy(Strategy):
    def __init__(self):
        Strategy.__init__(self,"Random")
    def compute_strategy(self,state,id_team,id_player):
        tools = MyState(state,id_team,id_player)
        return tools.aller(tools.ball_position())+ tools.shoot(tools.position_but())


class Fonceur(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Fonceur")
    def compute_strategy (self, state, id_team, id_player):
        if id_team == 2 :
            return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(0, GAME_HEIGHT/2)-state.ball.position)
        return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(GAME_WIDTH, GAME_HEIGHT/2)-state.ball.position)


class Attaquant(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Attaquant")
    def compute_strategy (self, state, id_team, id_player):
        if state.ball.position.x>GAME_WIDTH/4.0 :
            if id_team == 2 :
                return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(0, GAME_HEIGHT/2)-state.ball.position)
            return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(GAME_WIDTH, GAME_HEIGHT/2)-state.ball.position)
        return SoccerAction(Vector2D(GAME_WIDTH/2,GAME_HEIGHT/2)-state.player_state(id_team, id_player).position,Vector2D()) 
  
      
class Defenseur(Strategy):
    def __init__(self):
        Strategy.__init__(self, "Defenseur")
    def compute_strategy (self, state, id_team, id_player):
        if id_team == 1:
            if state.ball.position.x<(GAME_WIDTH/4.0) :
                return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(GAME_WIDTH, GAME_HEIGHT/2)-state.ball.position)
            return SoccerAction(Vector2D(GAME_WIDTH/16.0,GAME_HEIGHT/2)-state.player_state(id_team, id_player).position,Vector2D())     
        if state.ball.position.x>(GAME_WIDTH*(3.0/4)) :
            return SoccerAction(state.ball.position-state.player_state(id_team, id_player).position, Vector2D(0, GAME_HEIGHT/2)-state.ball.position)
        return SoccerAction(Vector2D(GAME_WIDTH*(15.0/16),GAME_HEIGHT/2)-state.player_state(id_team, id_player).position,Vector2D())     


## Creation d'une equipe
team1 = SoccerTeam(name="team1",login="etu1")
team2 = SoccerTeam(name="team2",login="etu2")
#team1.add("El Matador",Attaquant()) #Strategie attaquant
team1.add("O Monstro",Defenseur()) #Strategie defenseur
team2.add("LaMontagne",Defenseur())
#team2.add("Messi",Attaquant())
team1.add("Bipbip",Fonceur())
#team2.add("Testeur",TestStrategy())   #Strategie de test

#Creation d'une partie
simu = Simulation(team1,team2)
#Jouer et afficher la partie
show_simu(simu)
#show_state()
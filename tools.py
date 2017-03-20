#!/usr/bin/bash
# -*- coding: utf-8 -*

from soccersimulator.mdpsoccer import SoccerAction
from soccersimulator.utils import Vector2D
from soccersimulator.settings import GAME_HEIGHT, GAME_WIDTH, PLAYER_RADIUS, BALL_RADIUS

class Item(object):
    def __init__(self,state,id_team,id_player):
        self.state = state
        self.key = (id_team,id_player)
    
    @property
    def team_1(self):
        if self.key[0]==1:
            return True
        return False
        
    @property
    def team_2(self):
        if self.key[0]==2:
            return True
        return False
        
    @property
    def ball_position(self):
        """ vector2D"""
        return self.state.ball.position

    @property
    def ball_vitesse(self):
        """ vector2D"""
        return self.state.ball.vitesse
    
    @property
    def my_position(self):
        """renvoi la position d'un joueur sous la forme d'un couple (x,y)"""
        return self.state.player_state(self.key[0],self.key[1]).position
        
    @property
    def position_centre(self):
        return Vector2D(GAME_WIDTH/2,GAME_HEIGHT/2)
    
    @property
    def position_att(self):
        return Vector2D(GAME_WIDTH/2,GAME_HEIGHT/2+15)

    @property
    def position0(self):
        return Vector2D(0,0)
        
    @property
    def position_defenseur(self):
        if self.key[0] == 2 :
            return Vector2D(GAME_WIDTH*(15.0/16),GAME_HEIGHT/2)
        return Vector2D(GAME_WIDTH/16.0,GAME_HEIGHT/2)
    @property
    def position_defenseur2(self):
        if self.key[0] == 2 :
            return Vector2D(GAME_WIDTH*(15.0/16)+30,GAME_HEIGHT/2 +20)
        return Vector2D(GAME_WIDTH/16.0 +30,GAME_HEIGHT/2 + 20)
    

   

    @property  
    def equipier(self):
        """ renvoi l'etat du coequipier le plus proche"""
        l = [(i, j,self.state.player_state(i,j).position.distance(self.my_position)) for i, j in self.state.players if i==self.key[0] and j!=self.key[1]]
        
        idt, idp, dist = l[0]
        for i, j, d in l:
            if d < dist:
                idt, idp, dist = i, j, d
        return self.state.player_state(idt,idp)
    
    
    @property  
    def equipier_pos(self):
        """renvoi la position de l'équipier le plus proche"""
        l = [(i, j,self.state.player_state(i,j).position.distance(self.my_position)) for i, j in self.state.players if i==self.key[0] and j!=self.key[1]]
        
        idt, idp, dist = l[0]
        for i, j, d in l:
            if d < dist:
                idt, idp, dist = i, j, d
        return self.state.player_state(idt,idp).position
        
    @property
    def adversaire_proche(self):
        """renvoi l'état(playerstate) du joueur adverse le plus proche"""
        l = [(i, j,self.state.player_state(i,j).position.distance(self.my_position)) for i, j in self.state.players if i!=self.key[0] and j!=self.key[1]]
        
        idt, idp, dist = l[0]
        for i, j, d in l:
            if d < dist:
                idt, idp, dist = i, j, d
        return self.state.player_state(idt,idp)
        
    @property
    def pos_adversaire_proche(self):
        """renvoi la position du joueur adverse le plus proche"""
        l = [(i, j,self.state.player_state(i,j).position.distance(self.my_position)) for i, j in self.state.players if i!=self.key[0]]
        idt, idp, dist = l[0]
        for i, j, d in l:
            if d < dist:
                idt, idp, dist = i, j, d
        return self.state.player_state(idt,idp).position
    
                
    @property 
    def distance_ball(self):
        """renvoi la distance  entre un joueur et la balle"""
        return (self.ball_position-self.my_position).norm
    
    @property    
    def distance_player(self, p):
        """renvoi la distance  entre 2 joueurs sous la forme d'un nombre flottant"""
        return (p-self.my_position).norm
    
    @property              
    def vect_anticipe(self):    
        return self.state.ball.position+ 5.75*self.state.ball.vitesse
   
    @property              
    def vect_anticipe_att(self):    
        return self.state.ball.position+ 4.5*self.state.ball.vitesse    
    @property              
    def vect_anticipe_eq(self):    
        return self.equipier.position+ 51*self.equipier.vitesse
    
    @property
    def position_cage(self):
        if self.key[0]==2:
            return Vector2D(0,GAME_HEIGHT/2)
        return Vector2D(GAME_WIDTH,GAME_HEIGHT/2)
    
    @property
    def position_mes_cages(self):
        if self.key[0]==2:
            return Vector2D(GAME_WIDTH,GAME_HEIGHT/2)
        return Vector2D(0,GAME_HEIGHT/2)


    @property
    def can_shoot( self ) :
        if (self.ball_position-self.my_position).norm>= PLAYER_RADIUS + BALL_RADIUS:
                return False
        return True
    
    @property
    def distance_shoot(self):
        if (self.position_cage-self.my_position).norm<=45:
            return True
        return False
    
    @property
    def coups_denvoi(self):
        if self.ball_position.x==GAME_WIDTH/2: 
            if self.ball_position.y==GAME_HEIGHT/2:
                return True
        return False
    
    @property
    def can_attack(self):
        if self.key[0] == 1 :
            if self.ball_position.x>GAME_WIDTH/4.0 :
                return True
            return False
        if self.ball_position.x<GAME_WIDTH*(3.0/4) :
            return True
        return False
   
    @property
    def can_def(self):
        if self.key[0] == 1 :
            if self.ball_position.x<(GAME_WIDTH/4.0)+10 :
                return True
            return False
        if self.ball_position.x>(GAME_WIDTH*(3.0/4))-10 :
            return True
        return False

     
class Action(Item): 
    @property
    def shoot_but(self):
        if self.key[0] == 2 :
            return SoccerAction(Vector2D(),Vector2D(0, GAME_HEIGHT/2)-self.state.ball.position)
        return SoccerAction(Vector2D(),Vector2D(GAME_WIDTH, GAME_HEIGHT/2)-self.state.ball.position)
    
    def aller(self,p):
        return SoccerAction(p-self.my_position,Vector2D())
        
    def ralentir_p(self, p):
        return SoccerAction(p/250.0 -self.my_position/250.0,Vector2D())
   
    @property    
    def passe(self):
        return SoccerAction(Vector2D(),self.equipier_pos-self.ball_position)
    @property  
    def passe_vers(self,p):
        return SoccerAction(Vector2D(),p-self.ball_position)
        
     
    @property   
    def passe_coeq(self):
        return SoccerAction(Vector2D(),self.equipier_pos-self.ball_position)
    
    @property  
    def shoot_to(self, p):
        return SoccerAction(Vector2D(), p-self.ball_position)
    
    @property
    def immobile(self):
        return SoccerAction(Vector2D(), Vector2D())
    
    @property
    def tout_droit(self):
        if self.key[0]==2 :
            return SoccerAction(Vector2D(-2,0),Vector2D())
        return SoccerAction(Vector2D(2,0),Vector2D())
    
    @property
    def dribble(self):
        vec_pos = self.position_cage - self.ball_position
        return SoccerAction(Vector2D(),Vector2D( angle = vec_pos.angle, norm = vec_pos.norm/45 ))
    @property
    def dribble_v2(self):
        v = self.goal_adv-self.ball_position
        v = v.normalize()*1.68
        
    @property
    def dribble_solo(self):
        vec_pos = self.position_cage - self.ball_position
        return SoccerAction(Vector2D(),Vector2D( angle = vec_pos.angle, norm = vec_pos.norm/30 ))
    """ 
    @property
    def passe_angle(self):
        vec_pos = self.equipier_pos - self.ball_position
        return SoccerAction(Vector2D(),Vector2D( angle = vec_pos.angle, norm = vec_pos.norm/5))
    """
    @property
    def passe_anglev2(self):
        vec_pos= self.pos_equipier_prochev2 - self.ball_position
        return SoccerAction(Vector2D(),Vector2D( angle = vec_pos.angle, norm = vec_pos.norm/5))
        
     
    @property 
    def shoot_angle(self):
        vec_pos = self.position_cage - self.ball_position
        return SoccerAction(Vector2D(),Vector2D( angle = vec_pos.angle, norm = vec_pos.norm/8 ))
    @property     
    def aller_coeq(self):
        return self.aller(self.equipier_pos)

    @property
    def aller_vect(self):
        return SoccerAction(self.vect_anticipe-self.my_position,Vector2D())
        
    @property
    def aller_vect_att(self):
        return SoccerAction((self.vect_anticipe_att-self.my_position),Vector2D())
    
    @property
    def degagement(self):
        return SoccerAction(Vector2D(),self.position_att-self.ball_position)
    
 

class Strats(Action):
    
    @property
    def solo(self):
        if self.team_1 :
            if self.distance_shoot :
                if self.can_shoot :
                    return self.shoot_angle
            if self.can_shoot :
                    return self.dribble_solo
            return self.aller_vect
        if  self.distance_shoot:
            if self.can_shoot :
                return self.shoot_angle
        if self.can_shoot :
            return self.dribble_solo
        return self.aller_vect
        
    @property
    def fonce(self):
        if self.can_shoot :
            return self.shoot_but
        return self.aller(self.ball_position)
        
    @property
    def attaque(self):
        if self.can_attack :
            if  self.distance_shoot :
                if self.can_shoot :
                    return self.shoot_angle
            if self.can_shoot :
                return self.dribble
            return self.aller_vect_att
        return self.aller(self.position_att)
        
    @property
    def defense(self):
        if self.can_shoot :
            return self.degagement
        if self.can_def :
            return self.aller_vect
        return self.aller(self.position_defenseur)
        if self.ball.position.x>(GAME_WIDTH*(3.0/4))-10 :
            return self.aller_vect
        return self.aller(self.position_defenseur)
    
    @property
    def defense2(self):
        if self.can_shoot :
            return self.degagement
        if self.can_def :
            return self.aller_vect
        return self.aller(self.position_defenseur2)
        if self.ball.position.x>(GAME_WIDTH*(3.0/4))-10 :
            return self.aller_vect
        return self.aller(self.position_defenseur2)


    @property
    def defense_tutor(self):
        if self.can_shoot :
            return self.degagement
        if self.can_def :
            return self.aller_vect
        return self.aller(self.position_defenseur_tutor)
        if self.ball.position.x>(GAME_WIDTH*(3.0/4))-10 :
            return self.aller_vect
        return self.aller(self.position_defenseur_tutor)
    
    @property
    def defmilieu(self):
        if self.can_shoot :
            if self.distance_shoot:
                return self.shoot_angle
            return self.passe_coeq
        #si le joueur adverse est plus proche de la balle que le joueur actuel
        elif (self.position_mes_cages.distance(self.my_position)<70) or (self.pos_adversaire_proche.distance(self.ball_position)< self.my_position.distance(self.ball_position)):
            return self.aller(self.ball_position + (4*self.ball_vitesse) )
        else:
            return self.aller(self.position_mes_cages)
    
        
      
              



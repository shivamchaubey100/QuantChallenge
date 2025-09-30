"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional
import bisect
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0
class PriceLevel:
    def __init__(self, price: float, quantity: float = 0):
        self.price = price
        self.quantity = quantity

    def __repr__(self):
        return f"PriceLevel(price={self.price}, qty={self.quantity})"
    
from collections import defaultdict

class Team:
    def __init__(self, name: str):
        self.name = name
        self.points = 0
        self.shots_attempted = defaultdict(int)   # e.g., {"TWO_POINT": 0, "THREE_POINT": 0, "FREE_THROW": 0}
        self.shots_made = defaultdict(int)
        self.fouls = 0
        self.turnovers = 0
        self.rebounds = {"OFFENSIVE": 0, "DEFENSIVE": 0}
        self.steals = 0
        self.blocks = 0
    def reset_state(self,name):
        self.name = name
        self.points = 0
        self.shots_attempted = {"TWO_POINT": 0, "THREE_POINT": 0, "FREE_THROW": 0}
        self.shots_made =  {"TWO_POINT": 0, "THREE_POINT": 0, "FREE_THROW": 0}
        self.fouls = 0
        self.turnovers = 0
        self.rebounds = {"OFFENSIVE": 0, "DEFENSIVE": 0}
        self.steals = 0
        self.blocks = 0
    def __repr__(self):
        return f"<Team {self.name}: {self.points} pts>"

class Game:
    def __init__(self, home_team: str, away_team: str):
        self.home = Team(home_team)
        self.away = Team(away_team)
        self.period = 1
        self.time_remaining = None  # in seconds
        self.history = []  # store processed events for debugging/analysis
        self.possesion = None
        self.init_time=0
        self.init_time_set=False
        self.overtakes=0 #Number of times the lead changed
        self.lead=None #Which team is leading
    def reset_state(self,home_team,away_team):
        self.home.reset_state(home_team)
        self.away.reset_state(away_team)
        self.period = 1
        self.time_remaining = None
        self.history = []
        self.possesion = None
        self.init_time=0
        self.init_time_set=False
        self.overtakes=0  
        self.lead= None
    def update_history(self, event: dict):
        """Update game state based on an incoming event dict."""
        team = None
        if event["home_away"] == "home":
            team = self.home
            other_team=self.away
        elif event["home_away"] == "away":
            team = self.away
            other_team=self.home
        etype = event["event_type"]
        self.time_remaining = event["time_seconds"]

        if not self.init_time_set:
            self.init_time=event["time_seconds"]
            self.init_time_set=True

        if etype == "START_PERIOD":
            
            self.period += 1
        elif etype == "JUMP_BALL":
            self.possesion = event["home_away"]

        elif etype == "SCORE" and team:
            points = self._points_from_shot(event["shot_type"])
            team.points += points
            team.shots_attempted[self.to_string_point(event["shot_type"])] += 1
            team.shots_made[self.to_string_point(event["shot_type"])] += 1
            if event['home_away'] == 'away':
                self.possesion = 'home'
            else:
                self.possesion = 'away'


        elif etype == "MISSED" and team:
            team.shots_attempted[self.to_string_point(event["shot_type"])] += 1

        elif etype == "REBOUND" and team:
            team.rebounds[event["rebound_type"]] += 1
            self.possesion = event['home_away']
        elif etype == "FOUL" and team:
            team.fouls += 1

        elif etype == "TURNOVER" and team:
            team.turnovers += 1
            if event['home_away'] == 'away':
                self.possesion = 'home'
            else:
                self.possesion = 'away'

        elif etype == "STEAL" and team:
            team.steals += 1
            self.possesion = event['home_away']
        elif etype == "BLOCK" and team:
            team.blocks += 1

        elif etype in ["SUBSTITUTION", "TIMEOUT", "END_PERIOD"]:
            # You may log these separately if useful for prediction
            pass
        
        if event["home_score"]>event["away_score"]:
            if self.lead=="away":
                self.overtakes+=1
            self.lead="home"
        elif event["home_score"]<event["away_score"]:
            if self.lead=="home":
                self.overtakes+=1
            self.lead="away"
        
        # Save event for later analysis
        self.history.append(event)

    def _points_from_shot(self, shot_type: str) -> int:
        if shot_type == "TWO_POINT" or shot_type == "LAYUP" or shot_type == "DUNK":
            return 2
        elif shot_type == "THREE_POINT":
            return 3
        elif shot_type == "FREE_THROW":
            return 1
        return 0
    def to_string_point(self, shot_type: str) -> int:
        if shot_type == "TWO_POINT" or shot_type == "LAYUP" or shot_type == "DUNK":
            return "TWO_POINT"
        elif shot_type == "THREE_POINT":
            return "THREE_POINT"
        elif shot_type == "FREE_THROW":
            return "FREE_THROW"
        return ""

class OrderBook:
    def __init__(self):
        self.bids = []   # list of PriceLevel, sorted by price (descending)
        self.asks = []  # list of PriceLevel, sorted by price (ascending)
    def rst_state(self):
        self.bids = []
        self.asks = []
    def update_on_snapshot(self, bids, asks):
        self.bids = [PriceLevel(price, quantity) for price, quantity in bids]
        self.asks = [PriceLevel(price, quantity) for price, quantity in asks]
    def find_level(self, side: Side, price: float):
        '''
        Returns the level corresponding to a particular side and price'
        If found returns the level object else return None
        '''
        levels = self.bids if side == Side.BUY else self.asks
        for lvl in levels:
            if lvl.price == price: 
                return lvl
        return None

    def update_level(self, side: Side, price: float, qty: float):
        """
        Update (or insert) a price level.
        If qty=0, remove the price level if it exists.
        """
        levels = self.bids if side == Side.BUY else self.asks
        reverse = (side == Side.BUY)

        existing = self.find_level(side, price)  # use find_level, not _find_level
        if existing:
            if qty == 0:
                levels.remove(existing)
            else:
                existing.quantity = qty
        else:
            if qty > 0:
                new_level = PriceLevel(price, qty)
                keys = [-lvl.price if reverse else lvl.price for lvl in levels]
                idx = bisect.bisect_left(keys, -price if reverse else price)
                levels.insert(idx, new_level)

    def __repr__(self):
        book = "ORDER BOOK\n"
        book += "SELLS:\n"
        for o in self.asks:
            book += f"  {o}\n"
        book += "BUYS:\n"
        for o in self.bids:
            book += f"  {o}\n"
        return book
class Portfolio:
    def __init__(self, starting_capital: float):
        self.capital = starting_capital
        self.position = 0 # Price -> quantity
        self.avg_buy_price = 0
        self.avg_sell_price = 0
    def rst_state(self,starting_capital):
        self.position = 0 # Price -> quantity
        self.avg_buy_price = 0
        self.bought_qty = 0
        self.sold_qty = 0
        self.avg_sell_price = 0
    def update_position(self, ticker: Ticker, side: Side, quantity: float, price: float):
        """
        Update positions and capital after a trade.
        side: Side.BUY or Side.SELL
        quantity: number of shares traded
        price: trade price
        """
        # Update positions
        current_qty = self.position
        if side == Side.BUY:
            self.position = current_qty + quantity
            self.avg_buy_price =( (self.bought_qty*self.avg_buy_price) + (quantity*price))/(self.bought_qty+quantity)
            self.bought_qty += quantity
            self.capital -= quantity * price
        elif side == Side.SELL:
            self.position = current_qty - quantity
            self.avg_sell_price =( (self.sell_qty*self.avg_sell_price) + (quantity*price))/(self.sold_qty+quantity)
            self.sold_qty += quantity
            self.capital -= quantity * price
    def __repr__(self):
        return f"Portfolio(Capital: {self.capital}, Positions: {self.position}, Sold_qty: {self.sold_qty}, Avg_sell_Price:{self.avg_sell_price}, Bought_qty:{self.bought_qty},Avg Bought Price:{self.avg_buy_price})"
# theoretical price calculation
import numpy as np
import numpy as np
import math

class BinomialStrategy:
    def _init_(
        self,
        time_spent=1,
        time_remaining=0,
        home_team_attempted=0,
        away_team_attempted=0,
        home_team_shots_made=0,
        away_team_shots_made=0,
        home_team_score=0,
        away_team_score=0
    ):
    
        self.home_team_prob = (
            home_team_shots_made / home_team_attempted
            if home_team_attempted > 0
            else 0.5
        )
        self.away_team_prob = (
            away_team_shots_made / away_team_attempted
            if away_team_attempted > 0
            else 0.5
        )
     
        self.average_score_home_team = (
            float(home_team_score) / home_team_shots_made
            if home_team_attempted > 0
            else 0.0
        )
        self.average_score_away_team = (
            float(away_team_score) / away_team_shots_made
            if away_team_attempted > 0
            else 0.0
        )

        
        self.home_team_future_attempt = self._compute_future_attempts(
            time_spent, time_remaining, home_team_attempted
        )
        self.away_team_future_attempt = self._compute_future_attempts(
            time_spent, time_remaining, away_team_attempted
        )
        

    def _compute_future_attempts(self, time_spent: float, time_remaining: float, attempts_so_far: int) -> int:
      
        if time_remaining <= 0:
            return 0
        if time_spent != 0:
            rate = attempts_so_far / float(time_spent)
            est = rate * float(time_remaining)
            return max(0, int(round(est)))
        return 0

    def prob_update(
        self,
        time_spent: float,
        time_remaining: float,
        home_team_attempted: int,
        home_team_shots_made: int,
        away_team_attempted: int,
        away_team_shots_made: int,
        home_team_score: int,
        away_team_score: int
    ):
      
        self.home_team_prob = (
            home_team_shots_made / home_team_attempted if home_team_attempted > 0 else 0.5
        )
        self.away_team_prob = (
            away_team_shots_made / away_team_attempted if away_team_attempted > 0 else 0.5
        )

        self.average_score_home_team = (
            float(home_team_score) / home_team_shots_made if home_team_shots_made > 0 else 0.0
        )
        self.average_score_away_team = (
            float(away_team_score) / away_team_shots_made if away_team_shots_made > 0 else 0.0
        )

        self.home_team_future_attempt = self._compute_future_attempts(
            time_spent, time_remaining, home_team_attempted
        )
        self.away_team_future_attempt = self._compute_future_attempts(
            time_spent, time_remaining, away_team_attempted
        )


    def home_team_win_prob(self, home_team_score: int, away_team_score: int) -> float:
        """
        Compute probability that a*X + score_diff > b*Y,
        where X ~ Binomial(n_x, p_x), Y ~ Binomial(n_y, p_y),
        a = average_score_home_team, b = average_score_away_team,
        n_x = home_team_future_attempt, n_y = away_team_future_attempt.

        This implementation uses numpy for PMF/CDF calculations.
        """
        p_x = float(self.home_team_prob)
        p_y = float(self.away_team_prob)
        a = float(self.average_score_home_team)
        b = float(self.average_score_away_team)

        n_x = int(max(0, self.home_team_future_attempt))
        n_y = int(max(0, self.away_team_future_attempt))

        score_diff = int(home_team_score) - int(away_team_score)
        
        home_team_final_score = a*p_x*n_x + home_team_score

        away_team_final_score = b*p_y*n_y + away_team_score
        
        final_score_diff = home_team_final_score - away_team_final_score
        
        

        # # quick deterministic cases (no future attempts)
        # if n_x == 0 and n_y == 0:
        #     return 1.0 if score_diff > 0 else 0.0

        # eps = 1e-12
        # prob = 0.0

        # Precompute PMF and CDF for Y once using numpy arrays (if n_y > 0)
        # if n_y > 0:
        #     k_y = np.arange(0, n_y + 1)
        #     # compute combinatorial coefficients using math.comb in a vectorized list
        #     combs_y = np.array([math.comb(n_y, k) for k in k_y], dtype=float)
        #     pmf_y = combs_y * (p_y ** k_y) * ((1.0 - p_y) ** (n_y - k_y))
        #     cdf_y = np.cumsum(pmf_y)  # cdf_y[j] = P(Y <= j)
        # else:
        #     pmf_y = np.array([1.0])  # dummy
        #     cdf_y = np.array([1.0])

        # # Precompute PMF for X as an array
        # if n_x > 0:
        #     k_x = np.arange(0, n_x + 1)
        #     combs_x = np.array([math.comb(n_x, k) for k in k_x], dtype=float)
        #     pmf_x = combs_x * (p_x ** k_x) * ((1.0 - p_x) ** (n_x - k_x))
        # else:
        #     pmf_x = np.array([1.0])
        #     k_x = np.array([0])

        # # Iterate over x values (vectorized arrays) and accumulate probability
        # for idx, x in enumerate(k_x):
        #     px = float(pmf_x[idx])

        #     if abs(b) <= eps:
        #         # If b is zero, condition reduces to a*x + score_diff > 0
        #         win_if_x = (a * float(x) + score_diff) > 0
        #         prob_y = 1.0 if win_if_x else 0.0
        #     else:
        #         thresh = (a * float(x) + score_diff) / b
        #         # strict inequality: Y < thresh  -> Y <= floor(thresh - tiny)
        #         y_max = int(np.floor((thresh - 1e-12)))

        #         if y_max < 0:
        #             prob_y = 0.0
        #         elif n_y == 0:
        #             # no future Y attempts: Y is always 0
        #             prob_y = 1.0 if y_max >= 0 else 0.0
        #         elif y_max >= n_y:
        #             prob_y = 1.0
        #         else:
        #             prob_y = float(cdf_y[y_max])

        #     prob += px * prob_y

        print(f"home_score:{home_team_score}, away_score:{away_team_score}, n_x={n_x}, n_y={n_y}, p_x={p_x}, p_y={p_y}, a={a}, b={b}")
        # clamp numerical rounding
        # return float(min(1.0, max(0.0, prob)))

        return final_score_diff

class Strategy:
    """Template for a strategy."""
    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        self.ob.rst_state()
        self.portfolio.rst_state(100000)
        self.game.reset_state("home","away")
        self.binom_strat=BinomialStrategy()
        return 

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.ob = OrderBook()
        self.portfolio = Portfolio(100000)
        self.game = Game("home","away")
        self.binom_strat=BinomialStrategy()
        self.reset_state()
        return

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        self.ob.update_level(side,price,quantity)
        # print(self.ob)
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        print("-----Order Executed-----\n")
        print(f'Ticker-{ticker}   Side-{side}  at price ={price}  volume = {quantity} \n')
        print(f"------------- capital_remaining = {capital_remaining}-----------------------\n")
        self.portfolio.update_position(ticker,side,quantity,price)
        pass
    
    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
            """Called periodically with a complete snapshot of the orderbook.

            This provides the full current state of all bids and asks, useful for 
            verification and algorithms that need the complete market picture.

            Parameters
            ----------
            ticker
                Ticker of the orderbook snapshot (Ticker.TEAM_A)
            bids
                List of (price, quantity) tuples for all current bids, sorted by price descending
            asks  
                List of (price, quantity) tuples for all current asks, sorted by price ascending
            """
            self.ob.update_on_snapshot(bids,asks)
            return
    
    def trade_binom(self,estimate,threshold,confidence):
        current_pos=self.portfolio.position
        avg_buy_price=self.portfolio.avg_buy_price
        avg_sell_price=self.portfolio.avg_sell_price
        if self.ob.asks:
            best_ask=self.ob.asks[0].price
            best_ask_qaunt=self.ob.asks[0].quantity
        else:
            best_ask=estimate
            best_ask_qaunt=0
        if self.ob.bids:
            best_bid=self.ob.bids[0].price
            best_bid_qaunt=self.ob.bids[0].quantity
        else:
            best_bid=estimate
            best_ask_qaunt=0
        #CHECKING TO SEE IF WE ARE OVERLOADED
        if current_pos>confidence:
            place_market_order(Side.SELL,Ticker.TEAM_A,current_pos-confidence)
        elif current_pos<-confidence:
            place_market_order(Side.BUY,Ticker.TEAM_A,-current_pos-confidence)
        else:
            if best_bid<estimate-threshold:
                place_market_order(Side.BUY,Ticker.TEAM_A,5)
            elif best_ask>estimate+threshold:
                place_market_order(Side.SELL,Ticker.TEAM_A,5)
            else:
                if best_bid<estimate-2:
                    place_limit_order(Side.BUY,Ticker.TEAM_A,1,best_bid+1,ioc=True)
                if best_ask>estimate+2:
                    place_limit_order(Side.SELL,Ticker.TEAM_A,1,best_ask-1)
        return 

    from typing import List, Dict

    def execute_arbitrage(self, orderbook, price_cap: float = 100.0):    
        trades = []
        if not hasattr(orderbook, "bids") or not hasattr(orderbook, "asks"):
           return trades

        bids = orderbook.bids   
        asks = orderbook.asks  
       
        i = 0 
        j = 0  
     
        local_bids = [{"price": float(b.price), "remaining": float(b.quantity)} for b in bids]
        local_asks = [{"price": float(a.price), "remaining": float(a.quantity)} for a in asks]

        while i < len(local_bids) and j < len(local_asks):
            bid = local_bids[i]
            ask = local_asks[j]

            bid_price = bid["price"]
            ask_price = ask["price"]
            bid_rem = bid["remaining"]
            ask_rem = ask["remaining"]
      
            if bid_rem <= 0:
                i += 1
                continue
            if ask_rem <= 0:
                j += 1
                continue       
            if (bid_price + ask_price) >= price_cap:
                i += 1
                continue

            match_qty = min(bid_rem, ask_rem)
            if match_qty <= 0:
                if bid_rem <= 0:
                    i += 1
                if ask_rem <= 0:
                    j += 1
                continue        
            buy_order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, match_qty, ask_price, ioc=True)       
            sell_order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, match_qty, bid_price, ioc=True)        
            trades.append({
                "bid_index": i,
                "ask_index": j,
                "qty": match_qty,
                "buy_order_id": buy_order_id,
                "sell_order_id": sell_order_id,
                "buy_price": ask_price,
                "sell_price": bid_price
            })
            bid["remaining"] -= match_qty
            ask["remaining"] -= match_qty
            
            if bid["remaining"] <= 0:
                i += 1
            if ask["remaining"] <= 0:
                j += 1
        return trades
    
    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """
        if event_type == "NOTHING":
            return
        event={}
        event["event_type"]=event_type
        event["home_away"]=home_away
        event["home_score"]=home_score
        event["away_score"]=away_score
        event["player_name"]=player_name
        event["subsituted_player_name"]=substituted_player_name
        event["shot_type"]=shot_type
        event["assist_player"]=assist_player
        event["rebound_type"]=rebound_type
        event["coordinate_x"]=coordinate_x
        event["coordinate_y"]=coordinate_y
        event["time_seconds"]=time_seconds
        self.game.update_history(event)
        shot_attempted_home=sum(self.game.home.shots_attempted.values())+self.game.home.blocks
        shot_attempted_away=sum(self.game.away.shots_attempted.values())+self.game.away.blocks
        shots_made_home=sum(self.game.home.shots_made.values())
        shots_made_away=sum(self.game.away.shots_made.values())

        self.binom_strat.prob_update(self.game.init_time-time_seconds,
                                     time_seconds,
                                     shot_attempted_home,
                                     shots_made_home,
                                     shot_attempted_away,
                                     shots_made_away,
                                     self.game.home.points,
                                     self.game.away.points)
        estimate= self.binom_strat.home_team_win_prob(int(self.game.home.points),int(self.game.away.points))
        threshold=5
        # confidence=10
        # print(f'{time_seconds}::{event_type} {home_score} - {away_score}::Estimate::{estimate}')
        # self.trade_binom(estimate,threshold,confidence)
        print(f"time_remaining={time_seconds}")

        if(time_seconds < 800):
            if(estimate > 0):
                place_market_order(Side.BUY,Ticker.TEAM_A,1000)
            
            else:
                place_market_order(Side.SELL,Ticker.TEAM_A,1000)
        # else:
        #     self.execute_arbitrage(self.ob, 100.0)
        # if self.ob.bids and self.ob.bids[0].price>estimate+threshold:
        #     best_bid_price = self.ob.bids[0].price
        #     best_bid_volume = self.ob.bids[0].quantity
        #     place_market_order(Side.SELL,Ticker.TEAM_A,10)
        #     print("------------------------------------\n")
        #     print(f'Sell order placed at price = {best_bid_price}, volume = {1}\n')
        # if self.ob.asks and self.ob.asks[0].price<estimate-threshold:
        #     best_ask_price= self.ob.asks[0].price
        #     best_ask_volume=self.ob.asks[0].quantity
        #     place_market_order(Side.BUY,Ticker.TEAM_A,10)
        #     print("------------------------------------\n")
        #     print(f'Buy order placed at price = {best_ask_price}, volume = {1}\n') 
        # elif self.ob.bids and self.ob.asks:
        #     best_bid_price = self.ob.bids[0].price
        #     best_bid_volume = self.ob.bids[0].quantity
        #     best_ask_price= self.ob.asks[0].price
        #     best_ask_volume=self.ob.asks[0].quantity
        #     mid_price=(best_ask_price+best_bid_price)//2
        #     if estimate>mid_price+1:
        #         place_limit_order(Side.BUY,Ticker.TEAM_A,1,best_bid_price+1,ioc=True)
        #     elif estimate<mid_price-1:
        #         place_limit_order(Side.SELL,Ticker.TEAM_A,1,best_ask_price-1,ioc=True)
        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.reset_state()
            return
        
# Create strategy
# strat = Strategy()

# ob = OrderBook()
# ob.update_level(Side.BUY, 100, 10)
# ob.update_level(Side.SELL, 105, 20)
# ob.update_level(Side.BUY, 100, 20)   # remove BUY at 100
# ob.update_level(Side.BUY,90,100)
# ob.update_level(Side.SELL, 102, 20)
# print(ob)

# import sys
# import numpy as np
# sys.stdout = open("round1_game3.txt", "w")

# import json

# file = r"game3.json"
# with open(file, 'r') as f:
#     example_game = json.load(f)

# strat = Strategy()
# for event in example_game:
#     strat.on_game_event_update(**event)
# sys.stdout.close()

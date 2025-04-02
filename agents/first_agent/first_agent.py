import logging
import json
from decimal import Decimal
from os import path
from random import choice
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel

from typing import TypedDict

class SessionData(TypedDict):
    issue_weights: dict[str, float]
    value_counts: dict[str, dict[str, int]]
    utility_at_finish: float


def estimate_alpha_from_past_sessions(sessions: list[SessionData]) -> float:
    successful_sessions = [s for s in sessions if s["utility_at_finish"] > 0.0]
    if not successful_sessions:
        return 0.8  # fallback if previous negotiations always failed

    avg_util = sum(s["utility_at_finish"] for s in successful_sessions) / len(successful_sessions)

    # Tune alpha based on avg utility (such that alpha will be domain specific)
    return max(0.75, min(0.95, avg_util + 0.1))


class FirstAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None
        self.opponent_summary: dict[str, list[SessionData]] = None
        self.sorted_bids: list[Bid] = []
        self.alpha = 0.9 # default acceptance threshold
        self.utility_at_finish: float = 0.0 # utility of final agreement

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                if self.other is None:
                    # obtain the name of the opponent, cutting of the position ID.
                    self.other = str(actor).rsplit("_", 1)[0]
                    # setting alpha based on domain size if no previous session data available
                    self.alpha = self.estimate_alpha_from_domain_size()
                    self.load_data()

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            agreements = cast(Finished, data).getAgreements()
            # Save the utility of the accepted bid
            if len(agreements.getMap()) > 0:
                agreed_bid = agreements.getMap()[self.me]

                # Cast profile to LinearAdditiveUtilitySpace to access getUtility()
                linear_profile = cast(LinearAdditiveUtilitySpace, self.profile)
                self.utility_at_finish = float(linear_profile.getUtility(agreed_bid))
            else:
                self.utility_at_finish = 0 # No bid agreement

            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def load_data(self):
        filepath = f"{self.storage_dir}/{self.other}.json"
        domain_name = self.domain.getName()

        if path.exists(filepath):
            with open(filepath) as f:
                self.opponent_summary = json.load(f)
            if domain_name not in self.opponent_summary:
                self.opponent_summary[domain_name] = []
            else: # Calculate alpha based on the utility of the previously accepted bids
                self.alpha = estimate_alpha_from_past_sessions(self.opponent_summary[domain_name])
        else:
            self.opponent_summary = {domain_name: []}

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        if not self.storage_dir or not self.other:
            return

        if self.opponent_model:
            domain_name = self.domain.getName()
            if domain_name not in self.opponent_summary:
                self.opponent_summary[domain_name] = []

            session_summary = self.opponent_model.get_summary()
            session_summary["utility_at_finish"] = self.utility_at_finish
            self.opponent_summary[domain_name].append(session_summary)

            with open(f"{self.storage_dir}/{self.other}.json", "w") as f:
                f.write(json.dumps(self.opponent_summary, sort_keys=True, indent=4))

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def estimate_alpha_from_domain_size(self):
        domain_size = AllBidsList(self.domain).size()
        if domain_size < 3000:
            return 0.8  # small domain, be more cooperative
        elif domain_size < 6000:
            return 0.85  # medium domain
        else:
            return 0.9  # big domain, be conservative since the domain requires more exploration

    def get_dynamic_min_utility(self) -> float:
        if self.opponent_model and len(self.opponent_model.offers) >= 10:
            # Enough data to adapt based on opponent
            predicted_utils = [
                self.opponent_model.get_predicted_utility(bid)
                for bid in self.opponent_model.offers
            ]
            return max(0.6, min(0.85, sum(predicted_utils) / len(predicted_utils)))

        # If not enough opponent data, base it on domain size
        domain_size = AllBidsList(self.domain).size()
        if domain_size < 3000:
            return 0.6
        elif domain_size < 6000:
            return 0.7
        else:
            return 0.75

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)
        utility = self.profile.getUtility(bid)

        # More flexible as time progresses (with cap at 0.4)
        # dynamic_threshold = max(self.alpha - 0.6 * (progress ** 4), 0.4)
        dynamic_threshold = self.alpha / (1 + 5 * (progress ** 3))

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        # conditions = [
        #     self.profile.getUtility(bid) > 0.8,
        #     progress > 0.95,
        # ]

        if utility >= self.estimate_alpha_from_domain_size(): # Accept all very good offers for a specific domain
            return True

        if self.opponent_model is not None:
            predicted = self.opponent_model.get_predicted_utility(bid)
            if utility >= 0.7 and abs(utility - Decimal(str(predicted))) <= 0.2: # Accept if the received offer has a decent utility for us and it is favorable for both agents
                return True
        # return all(conditions)
        return utility >= dynamic_threshold

    def find_bid(self) -> Bid:
        # Decrease minimum acceptable utility value over time
        progress = self.progress.get(time() * 1000)
        if self.opponent_model and self.opponent_model.get_concession_speed() < 0.2:
            # Opponent is stubborn — delay more the decrease of the threshold
            min_util = self.get_dynamic_min_utility() - 0.4 * (progress ** 3)
        else:
            # Opponent concedes more — decrease faster
            min_util = self.get_dynamic_min_utility() - 0.8 * (progress ** 3)
        #min_util = self.get_dynamic_min_utility() - 0.6 * (progress ** 4)
        opponent_importance = progress ** 2  # Increase opponent influence as time progresses

        # compose a list of all possible bids and sort descending on utility
        if not self.sorted_bids:
            all_bids = AllBidsList(self.domain)
            self.sorted_bids = sorted(
                [all_bids.get(i) for i in range(all_bids.size())],
                key=lambda crt_bid: self.score_bid(crt_bid, alpha=1 - opponent_importance),
                reverse=True
            )

        top_n = 1  # Only pick best bid if the negotiation just started
        if progress >= 0.2:
            top_n = max(5, int(len(self.sorted_bids) * 0.01))

        top_bids = []
        for bid in self.sorted_bids:
            util = self.profile.getUtility(bid)
            if util < min_util:
                break

            score = self.score_bid(bid, 1 - opponent_importance)
            top_bids.append((bid, score))

            if len(top_bids) >= top_n:
                break

        if not top_bids:
            # Fallback: return the best possible bid below min_util
            for bid in self.sorted_bids:
                if self.profile.getUtility(bid) < min_util:
                    return bid

        # If we have at least one top-scoring bid, pick one randomly among them
        selected_bid = choice([bid for bid, _ in top_bids])
        target_utility = self.profile.getUtility(selected_bid)

        # Collect all bids with similar utility within epsilon range
        epsilon = 0.01
        similar_bids = [
            bid for bid in self.sorted_bids
            if abs(self.profile.getUtility(bid) - target_utility) <= epsilon
        ]

        # Choose the one with the highest opponent utility
        if self.opponent_model and similar_bids:
            return max(
                similar_bids,
                key=lambda b: self.opponent_model.get_predicted_utility(b)
                #key=lambda b: float(self.profile.getUtility(b)) * self.opponent_model.get_predicted_utility(b) --- using nash score
            )

        return selected_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score

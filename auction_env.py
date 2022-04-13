from message import Message

class Manager_Selection:
    def __init__(self, i, pattern, label, manager, agents, n_test_sample):
        #super().__init__()
        self.id = i
        self.manager = manager
        self.agents = agents
        self.pattern = pattern
        self.label = label
        self.team_rep = 0
        self.n_test_samples = n_test_sample

    def execute_auction(self):
        product_sold = False
        total_rep = 0
        no_of_participants = len(self.agents)
        print("******* Test Starts **********")
        print("Participants:")
        print("\tAuctioner: ", self.manager.name," on data #", str(self.id))
        print("\tBidders:")
        for n, agent in enumerate(self.agents):
            print("\t\tBidder ",n,"| reputation = ", agent.reputation)
            total_rep += agent.reputation
        self.team_rep = total_rep/no_of_participants
        print("Team Reputaion is " + str(self.team_rep))


            # broadcast message for start of auction
            # start_of_auction_msg = self.manager.broadcast_start_of_auction(self.id)
            # print("messageContent:", str(self.manager.name) + ":" + "auction of data #" + str(i) + " starts")

            # ask for proposal
        bids = []
        trust = []
        max_trust = 0
        most_trustable = None

        for n, agent in enumerate(self.agents):
            bids.append(agent.predict(self.pattern))
            trust.append(agent.reputation)
            if max_trust < agent.reputation:
                max_trust = agent.reputation
                most_trustable = n
        # callForProposal = Message(start_of_auction_msg, self.pattern, self.label, self.manager, self.agents,
        #                           "broadcast", "performative")

        # get proposed bids
        # broadCastbids = callForProposal.communicate()

            # select highest bid
            # bids_and_agents = {i: broadCastbids[i] for i in range(0, len(broadCastbids))}
        highest_bid = bids[most_trustable][0]

        # determine the winner(s)
        # winner_index = [i for i in range(len(bids_and_agents)) if bids_and_agents[i] == highest_bid]
        product_sold = True
        print("Highest bid of data #" + str(self.id) + " is ", highest_bid, " from agent ", most_trustable,
              "with reputation ", max_trust)

        # print("******* Updating Reputations **********")
        total_rep = 0
        for n, agent in enumerate(self.agents):
            total_rep += agent.update_reputation(self.n_test_samples, bids[n][0] == self.label[0])

        return (highest_bid, (total_rep/no_of_participants) + max_trust + bids[most_trustable][1])

        # print("WINNER ⭐️ ---The agent number " + str(
        #     winner_index) + " has won the auction for " + str(self.data_id))

        # broadcast message for end of auction
        # end_of_auction_msg = self.manager.broadcast_end_of_auction(self.id, most_trustable, highest_bid, "inform")
        # informAgents = Message(end_of_auction_msg, self.id, self.manager, self.agents, "broadcast",
        #                        "inform")
        # informAgents.inform()

        # update reputation
        # messageContent4 = self.manager.request_payment(winning_agent_index, highest_bid, "request")
        # requestAgentPayment = Message(messageContent4, self.auctioneer.product_id, self.auctioneer, self.bidders,
        #                               "message", "request")
        # requestAgentPayment.request(winning_agent_index)
        print("#" * 50)



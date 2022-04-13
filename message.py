class Message:
    def __init__(self, content, x, y, manager, agents, messageType, FCLprototype):
        self.content = content
        self.manager = manager
        self.agents = agents
        self.messageType = messageType
        self.FCLprototype = FCLprototype
        self.x = x
        self.y = y

    def communicate(self):
        if self.messageType == "broadcast":
            print("(" + str(self.manager.name) + " manager) " + "is sending " + self.messageType + "...")
        elif self.messageType == "message":
            print("(" + str(self.manager.name) + " manager) " + "is sending " + self.messageType + "...")

        bids = []

        contentSplit = self.content.split(":")

        # print("contentSplit",contentSplit)
        data_id = contentSplit[0]
        auctioned_content = contentSplit[1]
        fipa_protocol = contentSplit[2]

        if self.messageType == "message":
            print("sending --> " + auctioned_content)

        for agent in self.agents:
            bidders_bid = agent.interact(self.content, self.x, fipa_protocol)
            bids.append(bidders_bid)

        # print("bids",bids)
        # remember bis are [x,y]  where x is bid and y is bidder
        return bids

    def inform(self):
        if self.messageType == "broadcast":
            print("(" + str(self.manager.data_type) + " manager) " + "is sending " + self.messageType + "...")

        contentSplit = self.content.split(":")

        # print("contentSplit",contentSplit)
        highest_bid = contentSplit[0]
        # content = contentSplit[1]
        # winner  = contentSplit[2]
        fipa_protocol = contentSplit[3]

        for bidder in self.agents:
            # bidder.interact(self.content,auction_type,price,fipa_protocol)
            bidder.interact(self.content, highest_bid, fipa_protocol)

    def request(self, agent_id):
        if self.messageType == "message":
            print("(" + self.data_type + ")" + "sending " + self.messageType + "...")

        contentSplit = self.content.split(":")
        # print("contentSplit",contentSplit)
        highest_bid = contentSplit[0]
        # content = contentSplit[1]
        # winner  = contentSplit[2]
        fipa_protocol = contentSplit[3]

        # print ("agent_id",agent_id)
        self.bidders[agent_id].interact(self.content, "eng/dutch", highest_bid, fipa_protocol, agent_id)


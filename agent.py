import random

class Agent:
    def __init__(self, n, model, name):
        self.id = n
        self.name = name
        self.model = model
        self.reputation = 0
        self.output = None

    def train(self, X_train, Y_train, X_pred, Y_pred):
        # print("Training of Agent " + str(self.id) + " starts")
        if self.name == "FMM":
            self.model.train(X_train, Y_train)
        else:
            self.model.train()

        self.model.predict(X_pred, Y_pred)

        self.reputation = self.model.trust
        # print("Training of Agent " + str(self.id) + " ends")
        # print("Reputation of Agent " + str(self.id) + " is " + str(self.reputation))

    def predict(self,x):
        output, CF = self.model.test(x)
        return (output,CF)

    def update_reputation(self, n, correct):
        if correct:
            self.reputation *= (1 + self.reputation/n)
        else:
            self.reputation *= (1 - 1 / n)
        return self.reputation

    def interact(self, message, x, FIPAProtocol):

        if FIPAProtocol == "performative":
            self.model.test(x)
            return self.reputation

        elif FIPAProtocol == "inform":
            print("Agent " + str(self.id) + " received message -->" + message )

        # elif FIPAProtocol == "request":
        #     print("Transferring money to account")
        #     print("Purchase Ceiling-- ",self.price_ceiling)
        #     print("Initial Bank Balance-- ",self.bank_capacity)
        #     current_balance  = self.bank_capacity - int(currentPrice)
        #     current_utility  = self.price_ceiling - int(currentPrice)
        #     print("Amount  Paid-- ",currentPrice)
        #     print("Final Bank Balance-- ",current_balance)
        #     print("Utility (Initial Ceiling - Price bought) = ",current_utility)
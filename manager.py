class Manager:
    def __init__(self,model_name,reputation):
        self.name = model_name
        self.reputation = reputation

    def broadcast_start_of_auction(self,data_id):
        # add the ontology too ! "performative" , declarative, blahblha
        # product id, product price , prototype name,

        broadcast = str(self.name) + ":" + "auction of data #" + str(data_id) + " started :" + "performative"

        return broadcast

    def broadcast_end_of_auction (self,data_id, winner,amount,FIPAProtocol):
        #message = "The agent" + winner + "is the winner of this product"+ "with amount" + str(amount)
        broadcast = str(amount)+":" + "auction of " + str(data_id) + " completed : winner->"+"Agent "+str(winner)+":"+ FIPAProtocol
        return broadcast
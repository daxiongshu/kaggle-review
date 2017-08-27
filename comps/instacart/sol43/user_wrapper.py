class UserWrapper(object):
    NFEATS = 3 + 3 

    def __init__(self, user, mode="train"):
        """
        mode: train: 1) for all users 2) using default orders
              test: 1) only for test users 2) include testorder
              train_va: 1) for va users in train 2) get rid of the last order
              test_va: 1) for va users in train 2) using default orders
        """
        self.user = user
        self._all_pids = None
        self.mode = mode
        if mode == "test":
            assert self.user.test

    @property
    def orders(self):
        orders = self.user.orders
        if self.mode=="test":
            # Maybe cache this property if it's being accessed a lot?
            orders = list(orders)
            orders.append(self.user.testorder)
        #elif self.mode == "train_va":
        #    orders = list(orders)[:-1]
        return orders

    @property
    def seqlen(self):
        # Minus the first order, which is never a training example
        return len(self.orders) - 1

    @property
    def istest(self):
        return self.user.test 

    @property
    def uid(self):
        return self.user.uid

    @property
    def all_pids(self):
        """Return a set of ids of all products occurring in orders up to but not
        including the final one."""
        # This can get called a fair number of times, so cache it
        if self._all_pids is None:
            pids = set()
        else:
            return self._all_pids
        for order in self.orders[:-1]:
            pids.update( set(order.products) )
        self._all_pids = pids
        return self._all_pids

    @property
    def nprods(self):
        return len(self.all_pids)

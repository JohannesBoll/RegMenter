class Entity:
    """
    Parameters
    ----------
    tokenlist: list of tokens representing the entity
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    """
    def __init__(self, tokenlist, startid, endid):
        self.tokenlist = tokenlist
        self.startid = startid
        self.endid = endid

class Sentence(Entity):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing the sentence
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    """
    def __init__(self, tokenlist, startid, endid):
        super().__init__(tokenlist, startid, endid)

class List(Entity):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing the list
    introtokens: list of tokens representing the introductory sentence of a list
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    children: list of items representing the children of a list
    """
    def __init__(self, tokenlist, introtokens, startid, endid, children):
        super().__init__(tokenlist, startid, endid)
        self.introtokens = introtokens
        self.children = children

class Item(Entity):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing item
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    bulletlist: list of tokens representing a bulletpoint
    """
    def __init__(self, tokenlist, startid, endid, bulletlist):
        super().__init__(tokenlist, startid, endid)
        self.bulletlist = bulletlist

class Item1(Item):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing item1
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    children: list of item2 representing the child items
    bulletlist: list of tokens representing a bulletpoint
    """
    def __init__(self, tokenlist, startid, endid, children, bulletlist):

        super().__init__(tokenlist, startid, endid, bulletlist)
        self.children = children

class Item2(Item):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing item2
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    children: list of item3 representing the child items
    bulletlist: list of tokens representing a bulletpoint
    """
    def __init__(self, tokenlist, startid, endid, children, bulletlist):
        super().__init__(tokenlist, startid, endid, bulletlist)
        self.children = children

class Item3(Item):
    """
    Parameters
    ----------
    tokenlist: list of tokens representing the entity
    startid: relative startid in the document text string
    endid: relative startid in the document text string
    bulletlist: list of tokens representing a bulletpoint
    """
    def __init__(self, tokenlist, startid, endid, bulletlist):
        super().__init__(tokenlist, startid, endid, bulletlist)

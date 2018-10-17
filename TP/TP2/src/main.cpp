#include <bits/stdc++.h>

class Ant
{
public:
    Ant ();
    virtual ~Ant ();

private:
    int currV, nextV;
    std::vector<int> tabu;
    std::vector<int> path;

    double tourLength;
};

class Graph
{
public:
    Graph ();
    virtual ~Graph ();

private:
    std::unordered_map<int, int> edge;
};


Graph::Graph () {
    
}
Graph::~Graph () {
}

Ant::Ant() {
    std::cout << "Ant" << std::endl;
}
Ant::~Ant() {
    std::cout << "Desc Ant" << std::endl;
}
                          
int main()
{
    std::cout << "HelloWorld" << std::endl;
    return 0;
}

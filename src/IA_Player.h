#pragma once

#include <tuple>
#include <vector>
#include <memory>
#include <cassert>
#include <limits>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#include "Hex_Environement.h"

// Structure Union-Find pour une vérification de victoire ultra-rapide
class UnionFind {
    std::vector<int> parent;
public:
    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
        int rootI = find(i);
        int rootJ = find(j);
        if (rootI != rootJ) {
            parent[rootI] = rootJ;
        }
    }
};

// État de jeu simplifié pour accélérer les simulations MCTS
struct FastState {
    int size;
    std::vector<char> board;
    UnionFind uf;
    char current_player;
    int X_LEFT, X_RIGHT, O_TOP, O_BOTTOM;

    FastState(const Hex_Environement& env) : size(env.getSize()), board(size * size, '-'), uf(size * size + 4) {
        X_LEFT = size * size;
        X_RIGHT = size * size + 1;
        O_TOP = size * size + 2;
        O_BOTTOM = size * size + 3;
        current_player = env.getPlayer();

        // Initialisation de l'état avec le plateau actuel
        for(int r = 0; r < size; ++r) {
            for(int c = 0; c < size; ++c) {
                if (env.get(r, c) != '-') {
                    char piece = env.get(r, c);
                    board[r * size + c] = piece;
                    connectPiece(r, c, piece);
                }
            }
        }
    }

    void connectPiece(int r, int c, char piece) {
        int id = r * size + c;
        // Connexion aux bords virtuels
        if (piece == 'X') {
            if (c == 0) uf.unite(id, X_LEFT);
            if (c == size - 1) uf.unite(id, X_RIGHT);
        } else {
            if (r == 0) uf.unite(id, O_TOP);
            if (r == size - 1) uf.unite(id, O_BOTTOM);
        }

        // Connexion aux voisins de même couleur
        int dr[] = {-1, -1, 0, 0, 1, 1};
        int dc[] = {0, 1, -1, 1, -1, 0};
        for (int i = 0; i < 6; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                if (board[nr * size + nc] == piece) {
                    uf.unite(id, nr * size + nc);
                }
            }
        }
    }

    std::vector<int> getAvailableMoves() const {
        std::vector<int> moves;
        moves.reserve(size * size);
        for(int i = 0; i < size * size; ++i) {
            if(board[i] == '-') moves.push_back(i);
        }
        return moves;
    }

    void playMove(int id) {
        board[id] = current_player;
        int r = id / size;
        int c = id % size;
        connectPiece(r, c, current_player);
        current_player = (current_player == 'X') ? 'O' : 'X';
    }

    char getWinner() const {
        if (uf.find(X_LEFT) == uf.find(X_RIGHT)) return 'X';
        if (uf.find(O_TOP) == uf.find(O_BOTTOM)) return 'O';
        return '?';
    }
};

// Nœud de l'arbre MCTS
struct MCTSNode {
    int move; 
    char player_who_moved; 
    int wins;
    int visits;
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> untried_moves;

    MCTSNode(int move, char player_who_moved, MCTSNode* parent, std::vector<int> untried) 
        : move(move), player_who_moved(player_who_moved), wins(0), visits(0), parent(parent), untried_moves(std::move(untried)) {}

    MCTSNode* UCTSelectChild() {
        MCTSNode* best_child = nullptr;
        double best_score = -1.0;
        // Constante d'exploration UCT
        double C = std::sqrt(2.0);
        
        for (auto& child : children) {
            double exploit = (double)child->wins / child->visits;
            double explore = C * std::sqrt(std::log(visits) / child->visits);
            double score = exploit + explore;
            
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        return best_child;
    }
};

// L'Intelligence Artificielle
class IA_Player : public Player_Interface {
    char _player;
    unsigned int _taille;
public:
    IA_Player(char player, unsigned int taille=10) : _player(player), _taille(taille) {
        assert(player == 'X' || player == 'O');
    }

    void otherPlayerMove(int row, int col) override {
        // L'autre joueur a joué (row, col)
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        FastState root_state(hex);
        std::vector<int> possible_moves = root_state.getAvailableMoves();
        
        if (possible_moves.empty()) return {-1, -1};

        char root_player = root_state.current_player;
        char opponent = (root_player == 'X') ? 'O' : 'X';
        
        auto root = std::make_unique<MCTSNode>(-1, opponent, nullptr, possible_moves);

        auto start_time = std::chrono::steady_clock::now();
        std::mt19937 rng(std::random_device{}());

        // Limite fixée à 1.9 secondes pour garantir de ne pas dépasser les 2.0s
        while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() < 1900) {
            MCTSNode* node = root.get();
            FastState state = root_state;

            // Sélection
            while (node->untried_moves.empty() && !node->children.empty()) {
                node = node->UCTSelectChild();
                state.playMove(node->move);
            }

            // Expansion
            if (!node->untried_moves.empty() && state.getWinner() == '?') {
                std::uniform_int_distribution<int> dist(0, node->untried_moves.size() - 1);
                int idx = dist(rng);
                int move = node->untried_moves[idx];
                
                // Retrait du coup choisi
                node->untried_moves[idx] = node->untried_moves.back();
                node->untried_moves.pop_back();

                char next_player = state.current_player;
                state.playMove(move);
                
                auto new_node = std::make_unique<MCTSNode>(move, next_player, node, state.getAvailableMoves());
                node->children.push_back(std::move(new_node));
                node = node->children.back().get();
            }

            // Simulation
            std::vector<int> moves_to_simulate = state.getAvailableMoves();
            std::shuffle(moves_to_simulate.begin(), moves_to_simulate.end(), rng);
            
            int i = 0;
            while (state.getWinner() == '?' && i < moves_to_simulate.size()) {
                state.playMove(moves_to_simulate[i++]);
            }

            // Rétropropagation
            char winner = state.getWinner();
            while (node != nullptr) {
                node->visits++;
                if (winner == node->player_who_moved) {
                    node->wins++;
                }
                node = node->parent;
            }
        }

        // Choix du meilleur coup basé sur le mouvement le plus visité
        MCTSNode* best_child = nullptr;
        int max_visits = -1;
        for (auto& child : root->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                best_child = child.get();
            }
        }

        int best_move = best_child->move;
        return {best_move / _taille, best_move % _taille};
    }
};

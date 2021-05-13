#include <stdio.h>
#include "sys/time.h"
#include "string.h"
#include <stdbool.h>
#include <math.h>

#include <thrust/host_vector.h>

// All Constants, Globals, Enums

// Used for low level chess functions, such as adding a move, determining if a piece is under attack, etc.
// Some functions are used on the device, so they have a device version

enum side
{
    w,
    b
};
enum pieces
{
    emSq,
    wP,
    wN,
    wB,
    wR,
    wQ,
    wK,
    offBoard = 8,
    bP,
    bN,
    bB,
    bR,
    bQ,
    bK
};
enum castling
{
    K = 1,
    Q = 2,
    k = 4,
    q = 8
};

__device__ enum dev_castling { dev_K = 1,
                               dev_Q = 2,
                               dev_k = 4,
                               dev_q = 8 };

enum squares
{
    a1 = 0,
    b1,
    c1,
    d1,
    e1,
    f1,
    g1,
    h1,
    a2 = 16,
    b2,
    c2,
    d2,
    e2,
    f2,
    g2,
    h2,
    a3 = 32,
    b3,
    c3,
    d3,
    e3,
    f3,
    g3,
    h3,
    a4 = 48,
    b4,
    c4,
    d4,
    e4,
    f4,
    g4,
    h4,
    a5 = 64,
    b5,
    c5,
    d5,
    e5,
    f5,
    g5,
    h5,
    a6 = 80,
    b6,
    c6,
    d6,
    e6,
    f6,
    g6,
    h6,
    a7 = 96,
    b7,
    c7,
    d7,
    e7,
    f7,
    g7,
    h7,
    a8 = 112,
    b8,
    c8,
    d8,
    e8,
    f8,
    g8,
    h8,
    noSq = -99
};

enum moveFlags
{
    allPos,
    captures
};

// attack directions
const int pawnAttacks[4] = {15, 17, -15, -17};
const int knightAttacks[8] = {31, 33, 14, 18, -31, -33, -14, -18};
const int kingAttacks[8] = {1, 15, 16, 17, -1, -15, -16, -17};
const int bishopAttacks[4] = {15, 17, -15, -17};
const int rookAttacks[4] = {1, 16, -1, -16};

__device__ const int dev_pawnAttacks[4] = {15, 17, -15, -17};
__device__ const int dev_knightAttacks[8] = {31, 33, 14, 18, -31, -33, -14, -18};
__device__ const int dev_kingAttacks[8] = {1, 15, 16, 17, -1, -15, -16, -17};
__device__ const int dev_bishopAttacks[4] = {15, 17, -15, -17};
__device__ const int dev_rookAttacks[4] = {1, 16, -1, -16};

// tracking whether kings or rooks moved
const int castling[128] =
    {
        13, 15, 15, 15, 12, 15, 15, 14, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        7, 15, 15, 15, 3, 15, 15, 11, 8, 8, 8, 8, 8, 8, 8, 8};

__device__ const int dev_castling[128] =
    {
        13, 15, 15, 15, 12, 15, 15, 14, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8, 8, 8, 8,
        7, 15, 15, 15, 3, 15, 15, 11, 8, 8, 8, 8, 8, 8, 8, 8};

// material weight of pieces
const int materialWeight[15] =
    {
        0, 100, 300, 350, 525, 1000, 10000, 0,
        0, -100, -300, -350, -525, -1000, -10000};

__device__ const int dev_materialWeight[15] =
    {
        0, 100, 300, 350, 525, 1000, 10000, 0,
        0, -100, -300, -350, -525, -1000, -10000};

// piece placement tables, as pieces have different values depending on their position
const int Pawns[128] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 20, 20, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        20, 20, 20, 30, 30, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int Knights[128] =
    {
        0, -10, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 20, 20, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 15, 20, 20, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 20, 20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int Bishops[128] =
    {
        0, 0, -10, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 20, 20, 20, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 15, 20, 20, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int Rooks[128] =
    {
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        25, 25, 25, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int Kings[128] =
    {
        5, 5, 0, -10, -10, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 5, -10, -10, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 20, 20, 20, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 20, 20, 20, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Mirror evaluation tables for opposite side
const int Mirror[128] =
    {
        a8, b8, c8, d8, e8, f8, g8, h8, 0, 0, 0, 0, 0, 0, 0, 0,
        a7, b7, c7, d7, e7, f7, g7, h7, 0, 0, 0, 0, 0, 0, 0, 0,
        a6, b6, c6, d6, e6, f6, g6, h6, 0, 0, 0, 0, 0, 0, 0, 0,
        a5, b5, c5, d5, e5, f5, g5, h5, 0, 0, 0, 0, 0, 0, 0, 0,
        a4, b4, c4, d4, e4, f4, g4, h4, 0, 0, 0, 0, 0, 0, 0, 0,
        a3, b3, c3, d3, e3, f3, g3, h3, 0, 0, 0, 0, 0, 0, 0, 0,
        a2, b2, c2, d2, e2, f2, g2, h2, 0, 0, 0, 0, 0, 0, 0, 0,
        a1, b1, c1, d1, e1, f1, g1, h1, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Pawns[128] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 20, 20, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        20, 20, 20, 30, 30, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Knights[128] =
    {
        0, -10, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 20, 20, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 15, 20, 20, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 20, 20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Bishops[128] =
    {
        0, 0, -10, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 20, 20, 20, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 15, 20, 20, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Rooks[128] =
    {
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        25, 25, 25, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Kings[128] =
    {
        5, 5, 0, -10, -10, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 5, -10, -10, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 20, 20, 20, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 20, 20, 20, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const int dev_Mirror[128] =
    {
        a8, b8, c8, d8, e8, f8, g8, h8, 0, 0, 0, 0, 0, 0, 0, 0,
        a7, b7, c7, d7, e7, f7, g7, h7, 0, 0, 0, 0, 0, 0, 0, 0,
        a6, b6, c6, d6, e6, f6, g6, h6, 0, 0, 0, 0, 0, 0, 0, 0,
        a5, b5, c5, d5, e5, f5, g5, h5, 0, 0, 0, 0, 0, 0, 0, 0,
        a4, b4, c4, d4, e4, f4, g4, h4, 0, 0, 0, 0, 0, 0, 0, 0,
        a3, b3, c3, d3, e3, f3, g3, h3, 0, 0, 0, 0, 0, 0, 0, 0,
        a2, b2, c2, d2, e2, f2, g2, h2, 0, 0, 0, 0, 0, 0, 0, 0,
        a1, b1, c1, d1, e1, f1, g1, h1, 0, 0, 0, 0, 0, 0, 0, 0};

// Structs used for represending the board, search, and possible moves
typedef struct
{
    int move;
    int score;
} Move;
typedef struct
{
    Move moves[256];
    int moveCount;
} Movelist;

typedef struct
{
    int position[128];

    int side;
    int enPassant;
    int castle;

    int kingSq[2];

    int ply;
}

Chessboard;

typedef struct
{
    long nodes;
    double fhf, fh;

    int bestMove;
    int bestScore;

}

Search;

// Global vectors that hold all the boards and searches that get sent to the GPU
// They are global as I was having bugs with passing them by reference and security or missuse is not an issue here.

thrust::host_vector<Chessboard> boards;
thrust::host_vector<Search> searches;

// The following are a series of Macros that significantly clean up code presentation
// I may have gotten carried away ...

#define MirrorSq(sq) Mirror[sq]
#define DevMirrorSq(sq) dev_Mirror[sq]

// 0x88 math
#define IsOnBoard(sq) (!(sq & 0x88))
#define fr2sq(file, rank) (rank * 16 - file)
#define parse2sq(file, rank) ((rank - 1) * 16 + file)
#define GetFile(sq) (sq & 7)
#define GetRank(sq) (sq >> 4)
#define rank_7 (fromSq >= a7 && fromSq <= h7)
#define rank_2 (fromSq >= a2 && fromSq <= h2)

// Convertions
#define GetFileChar(sq) (GetFile(sq) + 'a')
#define GetRankChar(sq) (GetRank(sq) + '1')

// Char type
#define isDigit(char) (char >= '0' && char <= '9')
#define isPieceChar(piece) ((*fen >= 'a' && *fen <= 'z') || ((*fen >= 'A' && *fen <= 'Z')))

// White or black
#define isBlack(toSq) (board->position[toSq] >= bN && board->position[toSq] <= bQ)
#define isWhite(toSq) (board->position[toSq] >= wN && board->position[toSq] <= wQ)

#define isBlackPiece(toSq) (board->position[toSq] >= bP && board->position[toSq] <= bK)
#define DevisBlackPiece(toSq) (board->position[toSq] >= bP && board->position[toSq] <= bK)

#define isWhitePiece(toSq) (board->position[toSq] >= wP && board->position[toSq] <= wK)
#define DevisWhitePiece(toSq) (board->position[toSq] >= wP && board->position[toSq] <= wK)

// Quick board access
#define pos(sq) board->position[sq]
#define side board->side
#define enPassant board->enPassant
#define castle board->castle
#define kingSq(col) board->kingSq[col]
#define DevkingSq(col) board->kingSq[col]

#define ply board->ply

// Board loops
#define LoopBoard for (int sq = 0; sq < 128; ++sq)
#define RankLoop for (int rank = 8; rank >= 1; rank--)
#define FileLoop for (int file = 16; file >= 1; file--)

// Board methods
#define SetSq(sq, piece) (pos(sq) = piece)
#define DevSetSq(sq, piece) (pos(sq) = piece)

#define GetSq(sq) pos(sq)
#define DevGetSq(sq) pos(sq)

#define PrintSquare(sq) \
    if (sq == -99)      \
        printf("no");   \
    else                \
        printf("%c%c", GetFileChar(sq), GetRankChar(sq));
// #define PrintPromotedPiece(piece) printf("%c", promotedPieceChar[piece])

// Init board
#define ResetPosition(board) \
    LoopBoard { IsOnBoard(sq) ? SetSq(sq, emSq) : SetSq(sq, offBoard); }

#define ResetStats(board) \
    side = 0;             \
    enPassant = noSq;     \
    castle = 0;           \
    ply = 0;

#define ResetBoard(board) \
    ResetPosition(board); \
    ResetStats(board)

// Print board
#define PrintPosition(board)                              \
    printf("\n");                                         \
    RankLoop                                              \
    {                                                     \
        printf("  %d", rank);                             \
        FileLoop                                          \
        {                                                 \
            if (GetSq(fr2sq(file, rank)) != 8)            \
                printf("  %i", GetSq(fr2sq(file, rank))); \
        }                                                 \
        printf("\n");                                     \
    }

#define PrintStats(board)                                           \
    printf("\n     a  b  c  d  e  f  g  h\n\n");                    \
    printf("     Side:            %s\n", side ? "black" : "white"); \
    printf("     EnPassant:          ");                            \
    PrintSquare(enPassant);                                         \
    printf("\n");                                                   \
    printf("     Castling:         %c%c%c%c\n",                     \
           castle &K ? 'K' : '-',                                   \
           castle &Q ? 'Q' : '-',                                   \
           castle &k ? 'k' : '-',                                   \
           castle &q ? 'q' : '-');                                  \
    printf("\n\n");

#define PrintBoard(board) \
    PrintPosition(board); \
    PrintStats(board);

#define DevPrintPosition(board)                           \
    printf("\n");                                         \
    RankLoop                                              \
    {                                                     \
        printf("  %d", rank);                             \
        FileLoop                                          \
        {                                                 \
            if (GetSq(fr2sq(file, rank)) != 8)            \
                printf("  %i", GetSq(fr2sq(file, rank))); \
        }                                                 \
        printf("\n");                                     \
    }

#define DevPrintStats(dev_board)                                    \
    printf("\n     a  b  c  d  e  f  g  h\n\n");                    \
    printf("     Side:            %s\n", side ? "black" : "white"); \
    printf("     EnPassant:          ");                            \
    PrintSquare(enPassant);                                         \
    printf("\n");                                                   \
    printf("     Castling:         %c%c%c%c\n",                     \
           castle &K ? 'K' : '-',                                   \
           castle &Q ? 'Q' : '-',                                   \
           castle &k ? 'k' : '-',                                   \
           castle &q ? 'q' : '-');                                  \
    printf("\n\n");

#define DevPrintBoard(dev_board) \
    DevPrintPosition(dev_board); \
    DevPrintStats(dev_board);

#define SetMove(f, t, prom, cap, pawn, e, cas) \
    ((f) | (t << 7) | (prom << 14) | (cap << 18) | (pawn << 19) | (e << 20) | (cas << 21))
#define DevSetMove(f, t, prom, cap, pawn, e, cas) \
    ((f) | (t << 7) | (prom << 14) | (cap << 18) | (pawn << 19) | (e << 20) | (cas << 21))

#define GetMoveSource(move) (move & 0x7f)
#define DevGetMoveSource(move) (move & 0x7f)

#define GetMoveTarget(move) ((move >> 7) & 0x7f)
#define DevGetMoveTarget(move) ((move >> 7) & 0x7f)

#define GetMovePromPiece(move) ((move >> 14) & 0xf)
#define DevGetMovePromPiece(move) ((move >> 14) & 0xf)

#define GetMoveCaptureFlag(move) ((move >> 18) & 1)
#define DevGetMoveCaptureFlag(move) ((move >> 18) & 1)

#define GetMovePawnStartFlag(move) ((move >> 19) & 1)
#define DevGetMovePawnStartFlag(move) ((move >> 19) & 1)

#define GetMoveEnPassantFlag(move) ((move >> 20) & 1)
#define DevGetMoveEnPassantFlag(move) ((move >> 20) & 1)

#define GetMoveCastleFlag(move) ((move >> 21) & 1)
#define DevGetMoveCastleFlag(move) ((move >> 21) & 1)

#define SortMoves                                                            \
    for (int nextMove = moveNum + 1; nextMove < list->moveCount; ++nextMove) \
    {                                                                        \
        if (list->moves[moveNum].score < list->moves[nextMove].score)        \
        {                                                                    \
            int tempScore = list->moves[moveNum].score;                      \
            int tempMove = list->moves[moveNum].move;                        \
            list->moves[moveNum].score = list->moves[nextMove].score;        \
            list->moves[nextMove].score = tempScore;                         \
            list->moves[moveNum].move = list->moves[nextMove].move;          \
            list->moves[nextMove].move = tempMove;                           \
        }                                                                    \
    }

#define PrintMove(move)               \
    printf(" ");                      \
    PrintSquare(GetMoveSource(move)); \
    PrintSquare(GetMoveTarget(move));

#define LoopMoves for (int moveCount = 0; moveCount < list->moveCount; ++moveCount)

#define PrintMoveList(list)                                      \
    LoopMoves                                                    \
    {                                                            \
        PrintMove(list->moves[moveCount].move);                  \
        printf("	SCORE: %d\n", list->moves[moveCount].score); \
    }                                                            \
    printf("\n  Total moves: %d\n\n", list->moveCount);

#define TakeBack(board, boardStored) board[0] = boardStored[0];
#define DevTakeBack(board, boardStored) board[0] = boardStored[0];

#define InCheck(board, sideToMove) \
    IsSquareAttacked(board, sideToMove ? kingSq(b) : kingSq(w), sideToMove ^ 1)

#define DevInCheck(board, sideToMove) \
    DevIsSquareAttacked(board, sideToMove ? kingSq(b) : kingSq(w), sideToMove ^ 1)

/*****The real fun begins here!*****/

// Function that determines whether a piece on a given square is being attacked
// The function works by checking all possible positions that a given piece type can attack the given piece.
static inline int IsSquareAttacked(Chessboard *board, int sq, int attSide)
{
    // by pawns
    if (!attSide)
    {
        if (!((sq - 15) & 0x88) && (GetSq(sq - 15) == wP))
            return 1;

        if (!((sq - 17) & 0x88) && (GetSq(sq - 17) == wP))
            return 1;
    }

    else
    {
        if (!((sq + 15) & 0x88) && (GetSq(sq + 15) == bP))
            return 1;

        if (!((sq + 17) & 0x88) && (GetSq(sq + 17) == bP))
            return 1;
    }

    // by knights
    for (int i = 0; i < 8; ++i)
    {
        int dir = sq + knightAttacks[i];
        int delta = GetSq(dir);

        if (!(dir & 0x88))
        {
            if (attSide ? delta == bN : delta == wN)
                return 1;
        }
    }

    // by bishops and queens
    for (int i = 0; i < 4; ++i)
    {
        int dir = sq + bishopAttacks[i];

        while (!(dir & 0x88))
        {
            int delta = GetSq(dir);

            if (attSide ? (delta == bB) || (delta == bQ) : (delta == wB) || (delta == wQ))
                return 1;

            else if (delta != 0)
                break;

            dir += bishopAttacks[i];
        }
    }

    // by rooks and queens
    for (int i = 0; i < 4; ++i)
    {
        int dir = sq + rookAttacks[i];

        while (!(dir & 0x88))
        {
            int delta = GetSq(dir);

            if (attSide ? (delta == bR) || (delta == bQ) : (delta == wR) || (delta == wQ))
                return 1;

            else if (delta != 0)
                break;

            dir += rookAttacks[i];
        }
    }

    // by kings
    for (int i = 0; i < 8; ++i)
    {
        int dir = sq + kingAttacks[i];
        int delta = GetSq(dir);

        if (!(dir & 0x88))
        {
            if (attSide ? delta == bK : delta == wK)
                return 1;
        }
    }

    return 0;
}

// device version of the function above
__device__ static inline int DevIsSquareAttacked(Chessboard *board, int sq, int attSide)
{
    // by pawns
    if (!attSide)
    {
        if (!((sq - 15) & 0x88) && (GetSq(sq - 15) == wP))
            return 1;

        if (!((sq - 17) & 0x88) && (GetSq(sq - 17) == wP))
            return 1;
    }

    else
    {
        if (!((sq + 15) & 0x88) && (GetSq(sq + 15) == bP))
            return 1;

        if (!((sq + 17) & 0x88) && (GetSq(sq + 17) == bP))
            return 1;
    }

    // by knights
    for (int i = 0; i < 8; ++i)
    {
        int dir = sq + dev_knightAttacks[i];
        int delta = GetSq(dir);

        if (!(dir & 0x88))
        {
            if (attSide ? delta == bN : delta == wN)
                return 1;
        }
    }

    // by bishops and queens
    for (int i = 0; i < 4; ++i)
    {
        int dir = sq + dev_bishopAttacks[i];

        while (!(dir & 0x88))
        {
            int delta = GetSq(dir);

            if (attSide ? (delta == bB) || (delta == bQ) : (delta == wB) || (delta == wQ))
                return 1;

            else if (delta != 0)
                break;

            dir += dev_bishopAttacks[i];
        }
    }

    // by rooks and queens
    for (int i = 0; i < 4; ++i)
    {
        int dir = sq + dev_rookAttacks[i];

        while (!(dir & 0x88))
        {
            int delta = GetSq(dir);

            if (attSide ? (delta == bR) || (delta == bQ) : (delta == wR) || (delta == wQ))
                return 1;

            else if (delta != 0)
                break;

            dir += dev_rookAttacks[i];
        }
    }

    // by kings
    for (int i = 0; i < 8; ++i)
    {
        int dir = sq + dev_kingAttacks[i];
        int delta = GetSq(dir);

        if (!(dir & 0x88))
        {
            if (attSide ? delta == bK : delta == wK)
                return 1;
        }
    }

    return 0;
}

// Function that adds a move to our list of moves.
static inline void AddMove(Chessboard *board, Search *info, Movelist *list, int move)
{
    list->moves[list->moveCount].move = move;

    list->moveCount++;
}

// device version of the function above
__device__ static inline void DevAddMove(Chessboard *board, Search *info, Movelist *list, int move)
{
    //printf("here");

    list->moves[list->moveCount].move = move;

    list->moveCount++;
}

// Generates all possible moves, some of which are illegal
// The function works by looping through all squares and on each square checking
// the type of piece it is and then from there it adds all moves to our search and list
static inline void GenerateMoves(Chessboard *board, Search *info, Movelist *list)
{
    list->moveCount = 0;

    for (int sq = 0; sq < 128; ++sq)
    {
        if (!(sq & 0x88))
        {
            // skip empty squares
            if (!GetSq(sq))
                continue;

            int fromSq = sq;

            if (!side)
            {
                if (GetSq(fromSq) == wP)
                {
                    // pawn quiet move
                    if (!((fromSq + 16) & 0x88) && !GetSq(fromSq + 16))
                    {
                        if (rank_7 && !GetSq(fromSq + 16))
                        {
                            AddMove(board, info, list, SetMove(fromSq, fromSq + 16, wN, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq + 16, wB, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq + 16, wR, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq + 16, wQ, 0, 0, 0, 0));
                        }

                        else
                        {
                            AddMove(board, info, list, SetMove(fromSq, fromSq + 16, 0, 0, 0, 0, 0));

                            if (rank_2 && !GetSq(fromSq + 32))
                                AddMove(board, info, list, SetMove(fromSq, fromSq + 32, 0, 0, 1, 0, 0));
                        }
                    }

                    // pawn capture move
                    for (int i = 0; i < 4; ++i)
                    {
                        int dir = fromSq + pawnAttacks[i];

                        // en passant move
                        if (pawnAttacks[i] > 0 && !(dir & 0x88))
                        {
                            if (enPassant != noSq)
                            {
                                if (dir == enPassant)
                                    AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 1, 0));
                            }
                        }

                        if ((pawnAttacks[i] > 0) && !(dir & 0x88) && isBlackPiece(dir))
                        {
                            if (rank_7)
                            {
                                AddMove(board, info, list, SetMove(fromSq, dir, wN, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, wB, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, wR, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, wQ, 1, 0, 0, 0));
                            }

                            else
                            {
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            }
                        }
                    }
                }

                // castling
                if (GetSq(fromSq) == wK)
                {
                    if (castle & K)
                    {
                        if (!GetSq(f1) && !GetSq(g1))
                        {
                            if (!IsSquareAttacked(board, e1, b) && !IsSquareAttacked(board, f1, b))
                                AddMove(board, info, list, SetMove(e1, g1, 0, 0, 0, 0, 1));
                        }
                    }

                    if (castle & Q)
                    {
                        if (!GetSq(d1) && !GetSq(c1) && !GetSq(b1))
                        {
                            if (!IsSquareAttacked(board, e1, b) && !IsSquareAttacked(board, d1, b))
                                AddMove(board, info, list, SetMove(e1, c1, 0, 0, 0, 0, 1));
                        }
                    }
                }
            }

            else
            {
                if (GetSq(fromSq) == bP)
                {
                    // pawn quiet move
                    if (!((fromSq - 16) & 0x88) && !GetSq(fromSq - 16))
                    {
                        if (rank_2 && !GetSq(fromSq - 16))
                        {
                            AddMove(board, info, list, SetMove(fromSq, fromSq - 16, bN, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq - 16, bB, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq - 16, bR, 0, 0, 0, 0));
                            AddMove(board, info, list, SetMove(fromSq, fromSq - 16, bQ, 0, 0, 0, 0));
                        }

                        else
                        {
                            AddMove(board, info, list, SetMove(fromSq, fromSq - 16, 0, 0, 0, 0, 0));

                            if (rank_7 && !GetSq(fromSq - 32))
                                AddMove(board, info, list, SetMove(fromSq, fromSq - 32, 0, 0, 1, 0, 0));
                        }
                    }

                    // pawn capture move
                    for (int i = 0; i < 4; ++i)
                    {
                        int dir = fromSq + pawnAttacks[i];

                        // en passant move
                        if (pawnAttacks[i] < 0 && !(dir & 0x88))
                        {
                            if (enPassant != noSq)
                            {
                                if (dir == enPassant)
                                    AddMove(board, info, list, SetMove(fromSq, dir, 0, 0, 0, 1, 0));
                            }
                        }

                        if ((pawnAttacks[i] < 0) && !(dir & 0x88) && isWhitePiece(dir))
                        {
                            if (rank_2)
                            {
                                AddMove(board, info, list, SetMove(fromSq, dir, bN, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, bB, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, bR, 1, 0, 0, 0));
                                AddMove(board, info, list, SetMove(fromSq, dir, bQ, 1, 0, 0, 0));
                            }

                            else
                            {
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            }
                        }
                    }
                }

                // castling
                if (GetSq(fromSq) == bK)
                {
                    if (castle & k)
                    {
                        if (!GetSq(f8) && !GetSq(g8))
                        {
                            if (!IsSquareAttacked(board, e8, w) && !IsSquareAttacked(board, f8, w))
                                AddMove(board, info, list, SetMove(e8, g8, 0, 0, 0, 0, 1));
                        }
                    }

                    if (castle & q)
                    {
                        if (!GetSq(d8) && !GetSq(c8) && !GetSq(b8))
                        {
                            if (!IsSquareAttacked(board, e8, w) && !IsSquareAttacked(board, d8, w))
                                AddMove(board, info, list, SetMove(e8, c8, 0, 0, 0, 0, 1));
                        }
                    }
                }
            }

            // knights
            if (side ? GetSq(fromSq) == bN : GetSq(fromSq) == wN)
            {
                for (int i = 0; i < 8; ++i)
                {
                    int dir = sq + knightAttacks[i];
                    int delta = GetSq(dir);

                    if (!(dir & 0x88))
                    {
                        if (side ? (!delta || isWhitePiece(dir)) : (!delta || isBlackPiece(dir)))
                        {
                            if (!delta)
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 0, 0, 0, 0));
                            else
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                        }
                    }
                }
            }

            // bishops and queens
            if (side ? (GetSq(fromSq) == bB) || (GetSq(fromSq) == bQ) : (GetSq(fromSq) == wB) || (GetSq(fromSq) == wQ))

            {
                for (int i = 0; i < 4; ++i)
                {
                    int dir = sq + bishopAttacks[i];

                    while (!(dir & 0x88))
                    {
                        int delta = GetSq(dir);

                        // if hits own piece
                        if (side ? isBlackPiece(dir) : isWhitePiece(dir))
                            break;

                        // if hits opponent's piece
                        else if (side ? isWhitePiece(dir) : isBlackPiece(dir))
                        {
                            AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            break;
                        }

                        // on empty square
                        else if (!delta)
                        {
                            AddMove(board, info, list, SetMove(fromSq, dir, 0, 0, 0, 0, 0));
                        }

                        dir += bishopAttacks[i];
                    }
                }
            }

            // rooks and queens
            if (side ? (GetSq(fromSq) == bR) || (GetSq(fromSq) == bQ) : (GetSq(fromSq) == wR) || (GetSq(fromSq) == wQ))

            {
                for (int i = 0; i < 4; ++i)
                {
                    int dir = sq + rookAttacks[i];

                    while (!(dir & 0x88))
                    {
                        int delta = GetSq(dir);

                        // if hits own piece
                        if (side ? isBlackPiece(dir) : isWhitePiece(dir))
                            break;

                        // if hits opponent's piece
                        else if (side ? isWhitePiece(dir) : isBlackPiece(dir))
                        {
                            AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            break;
                        }

                        // on empty square
                        else if (!delta)
                        {
                            AddMove(board, info, list, SetMove(fromSq, dir, 0, 0, 0, 0, 0));
                        }

                        dir += rookAttacks[i];
                    }
                }
            }

            // kings
            if (side ? GetSq(fromSq) == bK : GetSq(fromSq) == wK)
            {
                for (int i = 0; i < 8; ++i)
                {
                    int dir = sq + kingAttacks[i];
                    int delta = GetSq(dir);

                    if (!(dir & 0x88))
                    {
                        if (side ? (!delta || isWhitePiece(dir)) : (!delta || isBlackPiece(dir)))
                        {
                            if (!delta)
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 0, 0, 0, 0));
                            else
                                AddMove(board, info, list, SetMove(fromSq, dir, 0, 1, 0, 0, 0));
                        }
                    }
                }
            }
        }
    }
}

// device version of the function above
__device__ static void DevGenerateMoves(Chessboard *board, Search *info, Movelist *list)
{
    list->moveCount = 0;
    //printf("some\n");

    for (int sq = 0; sq < 128; ++sq)
    {
        if (!(sq & 0x88))
        {
            // skip empty squares
            if (!DevGetSq(sq))
                continue;

            int fromSq = sq;

            if (!side)
            {
                if (DevGetSq(fromSq) == wP)
                {
                    // pawn quiet move
                    if (!((fromSq + 16) & 0x88) && !DevGetSq(fromSq + 16))
                    {
                        if (rank_7 && !DevGetSq(fromSq + 16))
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 16, wN, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 16, wB, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 16, wR, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 16, wQ, 0, 0, 0, 0));
                        }

                        else
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 16, 0, 0, 0, 0, 0));

                            if (rank_2 && !DevGetSq(fromSq + 32))
                                DevAddMove(board, info, list, DevSetMove(fromSq, fromSq + 32, 0, 0, 1, 0, 0));
                        }
                    }

                    // pawn capture move
                    for (int i = 0; i < 4; ++i)
                    {
                        int dir = fromSq + dev_pawnAttacks[i];

                        // en passant move
                        if (dev_pawnAttacks[i] > 0 && !(dir & 0x88))
                        {
                            if (enPassant != noSq)
                            {
                                if (dir == enPassant)
                                    DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 1, 0));
                            }
                        }

                        if ((dev_pawnAttacks[i] > 0) && !(dir & 0x88) && isBlackPiece(dir))
                        {
                            if (rank_7)
                            {
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, wN, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, wB, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, wR, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, wQ, 1, 0, 0, 0));
                            }

                            else
                            {
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            }
                        }
                    }
                }

                // castling
                if (DevGetSq(fromSq) == wK)
                {
                    if (castle & K)
                    {
                        if (!DevGetSq(f1) && !DevGetSq(g1))
                        {
                            if (!DevIsSquareAttacked(board, e1, b) && !DevIsSquareAttacked(board, f1, b))
                                DevAddMove(board, info, list, DevSetMove(e1, g1, 0, 0, 0, 0, 1));
                        }
                    }

                    if (castle & Q)
                    {
                        if (!DevGetSq(d1) && !DevGetSq(c1) && !DevGetSq(b1))
                        {
                            if (!DevIsSquareAttacked(board, e1, b) && !DevIsSquareAttacked(board, d1, b))
                                DevAddMove(board, info, list, DevSetMove(e1, c1, 0, 0, 0, 0, 1));
                        }
                    }
                }
            }

            else
            {
                if (DevGetSq(fromSq) == bP)
                {
                    // pawn quiet move
                    if (!((fromSq - 16) & 0x88) && !DevGetSq(fromSq - 16))
                    {
                        if (rank_2 && !DevGetSq(fromSq - 16))
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 16, bN, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 16, bB, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 16, bR, 0, 0, 0, 0));
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 16, bQ, 0, 0, 0, 0));
                        }

                        else
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 16, 0, 0, 0, 0, 0));

                            if (rank_7 && !DevGetSq(fromSq - 32))
                                DevAddMove(board, info, list, DevSetMove(fromSq, fromSq - 32, 0, 0, 1, 0, 0));
                        }
                    }

                    // pawn capture move
                    for (int i = 0; i < 4; ++i)
                    {
                        int dir = fromSq + dev_pawnAttacks[i];

                        // en passant move
                        if (dev_pawnAttacks[i] < 0 && !(dir & 0x88))
                        {
                            if (enPassant != noSq)
                            {
                                if (dir == enPassant)
                                    DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 0, 0, 1, 0));
                            }
                        }

                        if ((dev_pawnAttacks[i] < 0) && !(dir & 0x88) && isWhitePiece(dir))
                        {
                            if (rank_2)
                            {
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, bN, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, bB, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, bR, 1, 0, 0, 0));
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, bQ, 1, 0, 0, 0));
                            }

                            else
                            {
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            }
                        }
                    }
                }

                // castling
                if (DevGetSq(fromSq) == bK)
                {
                    if (castle & k)
                    {
                        if (!DevGetSq(f8) && !DevGetSq(g8))
                        {
                            if (!DevIsSquareAttacked(board, e8, w) && !DevIsSquareAttacked(board, f8, w))
                                DevAddMove(board, info, list, DevSetMove(e8, g8, 0, 0, 0, 0, 1));
                        }
                    }

                    if (castle & q)
                    {
                        if (!DevGetSq(d8) && !DevGetSq(c8) && !DevGetSq(b8))
                        {
                            if (!DevIsSquareAttacked(board, e8, w) && !DevIsSquareAttacked(board, d8, w))
                                DevAddMove(board, info, list, DevSetMove(e8, c8, 0, 0, 0, 0, 1));
                        }
                    }
                }
            }

            // knights
            if (side ? DevGetSq(fromSq) == bN : DevGetSq(fromSq) == wN)
            {
                for (int i = 0; i < 8; ++i)
                {
                    int dir = sq + dev_knightAttacks[i];
                    int delta = DevGetSq(dir);

                    if (!(dir & 0x88))
                    {
                        if (side ? (!delta || isWhitePiece(dir)) : (!delta || isBlackPiece(dir)))
                        {
                            if (!delta)
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 0, 0, 0, 0));
                            else
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                        }
                    }
                }
            }

            // bishops and queens
            if (side ? (DevGetSq(fromSq) == bB) || (DevGetSq(fromSq) == bQ) : (DevGetSq(fromSq) == wB) || (DevGetSq(fromSq) == wQ))

            {
                for (int i = 0; i < 4; ++i)
                {
                    int dir = sq + dev_bishopAttacks[i];

                    while (!(dir & 0x88))
                    {
                        int delta = DevGetSq(dir);

                        // if hits own piece
                        if (side ? isBlackPiece(dir) : isWhitePiece(dir))
                            break;

                        // if hits opponent's piece
                        else if (side ? isWhitePiece(dir) : isBlackPiece(dir))
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            break;
                        }

                        // on empty square
                        else if (!delta)
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 0, 0, 0, 0));
                        }

                        dir += dev_bishopAttacks[i];
                    }
                }
            }

            // rooks and queens
            if (side ? (DevGetSq(fromSq) == bR) || (DevGetSq(fromSq) == bQ) : (DevGetSq(fromSq) == wR) || (DevGetSq(fromSq) == wQ))

            {
                for (int i = 0; i < 4; ++i)
                {
                    int dir = sq + dev_rookAttacks[i];

                    while (!(dir & 0x88))
                    {
                        int delta = DevGetSq(dir);

                        // if hits own piece
                        if (side ? isBlackPiece(dir) : isWhitePiece(dir))
                            break;

                        // if hits opponent's piece
                        else if (side ? isWhitePiece(dir) : isBlackPiece(dir))
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                            break;
                        }

                        // on empty square
                        else if (!delta)
                        {
                            DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 0, 0, 0, 0));
                        }

                        dir += dev_rookAttacks[i];
                    }
                }
            }

            // kings
            if (side ? DevGetSq(fromSq) == bK : DevGetSq(fromSq) == wK)
            {
                for (int i = 0; i < 8; ++i)
                {
                    int dir = sq + dev_kingAttacks[i];
                    int delta = DevGetSq(dir);

                    if (!(dir & 0x88))
                    {
                        if (side ? (!delta || isWhitePiece(dir)) : (!delta || isBlackPiece(dir)))
                        {
                            if (!delta)
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 0, 0, 0, 0));
                            else
                                DevAddMove(board, info, list, DevSetMove(fromSq, dir, 0, 1, 0, 0, 0));
                        }
                    }
                }
            }
        }
    }
}

// Function that makes a Move and returns a number based on whether or not the move is legal
// The function works by taking the move and checking whether or not its legal, and if it is
// then the function will change the board position
static inline int MakeMove(Chessboard *board, int move, int capFlag)
{
    // if capFlag make only captures else make all

    if (!capFlag)
    {
        ply++;

        Chessboard boardStored[1];
        boardStored[0] = board[0];

        int fromSq = GetMoveSource(move);
        int toSq = GetMoveTarget(move);

        // move piece
        GetSq(toSq) = GetSq(fromSq);
        GetSq(fromSq) = emSq;

        // promotions
        if (GetMovePromPiece(move))
        {
            GetSq(toSq) = GetMovePromPiece(move);
            GetSq(fromSq) = emSq;
        }

        // en passant flag
        if (GetMoveEnPassantFlag(move))
        {
            side ? (GetSq(enPassant + 16) = 0) : (GetSq(enPassant - 16) = 0);

            enPassant = noSq;
        }

        enPassant = noSq;

        // pawn start flag
        if (GetMovePawnStartFlag(move))
        {
            side ? (enPassant = toSq + 16) : (enPassant = toSq - 16);
        }

        // castling flag
        if (GetMoveCastleFlag(move))
        {
            switch (toSq)
            {
            case g1:
                GetSq(f1) = GetSq(h1);
                GetSq(h1) = emSq;
                break;

            case c1:
                GetSq(d1) = GetSq(a1);
                GetSq(a1) = emSq;
                break;

            case g8:
                GetSq(f8) = GetSq(h8);
                GetSq(h8) = emSq;
                break;

            case c8:
                GetSq(d8) = GetSq(a8);
                GetSq(a8) = emSq;
                break;
            }
        }

        // update castling permission
        castle &= castling[fromSq];
        castle &= castling[toSq];

        // update kingSq
        if (GetSq(GetMoveTarget(move)) == wK || GetSq(GetMoveTarget(move)) == bK)
            kingSq(side) = GetMoveTarget(move);

        // change side
        side ^= 1;

        // take back if king is in check
        if (InCheck(board, side ^ 1))
        {
            TakeBack(board, boardStored);
            return 0;
        }

        else
            return 1;
    }

    else
    {
        if (GetMoveCaptureFlag(move))
            MakeMove(board, move, allPos);
        else
            return 0;
    }

    return 0;
}

// Device version of function above
__device__ static inline int DevMakeMove(Chessboard *board, int move, int capFlag)
{
    // if capFlag make only captures else make all

    if (!capFlag)
    {
        ply++;

        Chessboard boardStored[1];
        boardStored[0] = board[0];

        int fromSq = DevGetMoveSource(move);
        int toSq = DevGetMoveTarget(move);

        // move piece
        DevGetSq(toSq) = DevGetSq(fromSq);
        DevGetSq(fromSq) = emSq;

        // promotions
        if (DevGetMovePromPiece(move))
        {
            DevGetSq(toSq) = DevGetMovePromPiece(move);
            DevGetSq(fromSq) = emSq;
        }

        // en passant flag
        if (DevGetMoveEnPassantFlag(move))
        {
            side ? (DevGetSq(enPassant + 16) = 0) : (DevGetSq(enPassant - 16) = 0);

            enPassant = noSq;
        }

        enPassant = noSq;

        // pawn start flag
        if (DevGetMovePawnStartFlag(move))
        {
            side ? (enPassant = toSq + 16) : (enPassant = toSq - 16);
        }

        // castling flag
        if (DevGetMoveCastleFlag(move))
        {
            switch (toSq)
            {
            case g1:
                DevGetSq(f1) = DevGetSq(h1);
                DevGetSq(h1) = emSq;
                break;

            case c1:
                DevGetSq(d1) = DevGetSq(a1);
                DevGetSq(a1) = emSq;
                break;

            case g8:
                DevGetSq(f8) = DevGetSq(h8);
                DevGetSq(h8) = emSq;
                break;

            case c8:
                DevGetSq(d8) = DevGetSq(a8);
                DevGetSq(a8) = emSq;
                break;
            }
        }

        // update castling permission
        castle &= dev_castling[fromSq];
        castle &= dev_castling[toSq];

        // update kingSq
        if (DevGetSq(DevGetMoveTarget(move)) == wK || DevGetSq(DevGetMoveTarget(move)) == bK)
            DevkingSq(side) = DevGetMoveTarget(move);

        // change side
        side ^= 1;

        // take back if king is in check
        if (DevInCheck(board, side ^ 1))
        {
            DevTakeBack(board, boardStored);
            return 0;
        }

        else
            return 1;
    }

    else
    {
        if (DevGetMoveCaptureFlag(move))
            DevMakeMove(board, move, allPos);
        else
            return 0;
    }

    return 0;
}

// Evaluates the position
// The function loops through all the pieces on the board and
// adds the material weight of every piece to a sum
// the material weight is defined by the tables at the top
// which are the stockfish analysis tables.
static inline int EvaluatePosition(Chessboard *board)
{
    int score = 0;

    for (int sq = 0; sq < 128; ++sq)
    {
        if (!(sq & 0x88) && GetSq(sq))
        {
            // evaluate material
            score += materialWeight[GetSq(sq)];

            // evaluate piece placement
            switch (GetSq(sq))
            {
            case wP:
                score += Pawns[sq];
                break;

            case wN:
                score += Knights[sq];
                break;

            case wB:
                score += Bishops[sq];
                break;

            case wR:
                score += Rooks[sq];
                break;

            case wK:
                score += Kings[sq];
                break;

            case bP:
                score -= Pawns[MirrorSq(sq)];
                break;

            case bN:
                score -= Knights[MirrorSq(sq)];
                break;

            case bB:
                score -= Bishops[MirrorSq(sq)];
                break;

            case bR:
                score -= Rooks[MirrorSq(sq)];
                break;

            case bK:
                score -= Kings[MirrorSq(sq)];
                break;
            }
        }
    }

    if (!side)
        return score;

    else
        return -score;
}

// Device repeat function
__device__ static inline int DevEvaluatePosition(Chessboard *board)
{
    int score = 0;

    for (int sq = 0; sq < 128; ++sq)
    {
        if (!(sq & 0x88) && GetSq(sq))
        {
            // evaluate material
            score += dev_materialWeight[GetSq(sq)];

            // evaluate piece placement
            switch (GetSq(sq))
            {
            case wP:
                score += dev_Pawns[sq];
                break;

            case wN:
                score += dev_Knights[sq];
                break;

            case wB:
                score += dev_Bishops[sq];
                break;

            case wR:
                score += dev_Rooks[sq];
                break;

            case wK:
                score += dev_Kings[sq];
                break;

            case bP:
                score -= dev_Pawns[DevMirrorSq(sq)];
                break;

            case bN:
                score -= dev_Knights[DevMirrorSq(sq)];
                break;

            case bB:
                score -= dev_Bishops[DevMirrorSq(sq)];
                break;

            case bR:
                score -= dev_Rooks[DevMirrorSq(sq)];
                break;

            case bK:
                score -= dev_Kings[DevMirrorSq(sq)];
                break;
            }
        }
    }

    if (!side)
        return score;

    else
        return -score;
}

// Sets all of the Search variables to 0
void InitSearch(Search *info)
{
    info->nodes = 0;
    info->fhf = 0;
    info->fh = 0;
    info->bestScore = 0;
}

// NegaMaxSearch that runs entirely on the CPU, for comparison purposes, explained in doc
static int RegNegaMaxSearch(Chessboard *board, Search *info, int depth)
{
    int bestMove = 0;
    int alpha = -50000;
    int oldAlpha = alpha;
    int score = -50000;
    int legalMoves = 0;

    info->nodes++;

    if (depth == 0)
        return EvaluatePosition(board);

    Movelist list[1];
    GenerateMoves(board, info, list);

    // loops through all possible moves, recurssively calls function
    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        Chessboard boardStored[1];
        boardStored[0] = board[0];

        SortMoves;

        if (!MakeMove(board, list->moves[moveNum].move, allPos))
            continue;

        legalMoves++;
        score = -RegNegaMaxSearch(board, info, depth - 1);
        TakeBack(board, boardStored);

        if (score > alpha)
        {
            alpha = score;
            bestMove = list->moves[moveNum].move;
        }
    }

    if (!legalMoves)
    {
        if (InCheck(board, side))
            return -49000 + ply; // on checkmate

        else
            return 0; // on stalemate
    }

    if (alpha != oldAlpha)
    {
        info->bestMove = bestMove;
    }

    return alpha;
}

// NegaMaxSearch used when CPU depth is above 2, function is not in use anywhere as CPUdepth > 2 is not allowed
static int EvalNegaMaxSearch(Chessboard *board, Search *info, Search *searchP, int depth, int &count)
{
    int bestMove = 0;
    int alpha = -50000;
    int oldAlpha = alpha;
    int score = -50000;
    int legalMoves = 0;

    info->nodes++;

    if (depth == 0)
    {
        count++;
        return searchP[count - 1].bestScore;
    }

    Movelist list[1];
    GenerateMoves(board, info, list);

    // the good loop
    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        Chessboard boardStored[1];
        boardStored[0] = board[0];

        SortMoves;

        if (!MakeMove(board, list->moves[moveNum].move, allPos))
            continue;

        legalMoves++;
        score = -EvalNegaMaxSearch(board, info, searchP, depth - 1, count);
        TakeBack(board, boardStored);

        if (score > alpha)
        {
            alpha = score;
            bestMove = list->moves[moveNum].move;
        }
    }

    if (!legalMoves)
    {
        if (InCheck(board, side))
            return -49000 + ply; // on checkmate

        else
            return 0; // on stalemate
    }

    if (alpha != oldAlpha)
    {
        info->bestMove = bestMove;
    }

    return alpha;
}

// Same as CreateNegaMaxSearch except dcount is no longer here and since we aren't at the top depth validMoves are not being added
static void ContinueNegaMaxSearch(Chessboard *board, Search *info, int depth, int &count)
{

    info->nodes++;

    if (depth == 0)
    {

        Search push;
        InitSearch(&push);

        boards[count] = board[0];
        searches.push_back(push);

        count++;

        return;
    }

    Movelist list[1];

    GenerateMoves(board, info, list);

    // the good loop
    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        Chessboard boardStored[1];
        boardStored[0] = board[0];

        if (!MakeMove(board, list->moves[moveNum].move, allPos))
            continue;

        ContinueNegaMaxSearch(board, info, depth - 1, count);
        // store the count in a vector

        TakeBack(board, boardStored);
    }

    return;
}

//NegaMaxSearch style searchwith the adding of the board positions and info to the vectors.
static void CreateNegaMaxSearch(Chessboard *board, Search *info, Move *valid, int *moveCounterPtr, int depth, int &count, int &dcount)
{

    info->nodes++;

    if (depth == 0)
    {

        Search push;
        InitSearch(&push);

        boards[count] = board[0];
        searches.insert(searches.begin(), push);

        count++;
        return;
    }

    Movelist list[1];

    GenerateMoves(board, info, list);

    // Loops through all possible moves at this level and recursively calls itself with 1 less depth
    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        Chessboard boardStored[1];
        boardStored[0] = board[0];

        if (!MakeMove(board, list->moves[moveNum].move, allPos))
            continue;

        // Adds valid moves to array for analysis after GPU search
        valid[dcount].move = list->moves[moveNum].move;

        ContinueNegaMaxSearch(board, info, depth - 1, count);

        // Counter used for searching when CPUdepth = 2 to do a MiniMax search
        moveCounterPtr[dcount] = count;
        //moveCounter->push_back(count[0]);
        dcount++;

        TakeBack(board, boardStored);
    }
}

// NegaMaxSearch that the GPU should call, Same as RegNegaMax
__device__ static int SplitNegaMaxSearch(Chessboard *board, Search *info, int depth)
{

    int bestMove = 0;
    int alpha = -50000;
    int oldAlpha = alpha;
    int score = -50000;
    int legalMoves = 0;

    info->nodes++;

    if (depth == 0)
    {
        //printf("ep ");
        return DevEvaluatePosition(board);
    }

    Movelist list[1];

    DevGenerateMoves(board, info, list);

    // the good loop
    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        Chessboard boardStored[1];
        boardStored[0] = board[0];

        SortMoves;

        if (!DevMakeMove(board, list->moves[moveNum].move, allPos))
            continue;

        legalMoves++;
        score = -SplitNegaMaxSearch(board, info, depth - 1);

        // this feels really inefficient as we could just DevMakeMove the same move backward
        DevTakeBack(board, boardStored);

        if (score > alpha)
        {
            alpha = score;
            bestMove = list->moves[moveNum].move;
        }
    }

    if (!legalMoves)
    {
        if (DevInCheck(board, side))
            return -49000 + ply; // on checkmate

        else
            return 0; // on stalemate
    }

    if (alpha != oldAlpha)
    {
        info->bestMove = bestMove;
        info->bestScore = alpha;
    }

    return alpha;
}

// Kernel function called that starts searching assuming the thread number is in our needed array of searches
__global__ static void kernelSearch(Chessboard *dev_board, Search *dev_searches, int *dev_totalThreadCount, int *gpu_depth)
{

    int location = 256 * blockIdx.x + threadIdx.x;
    int depth = gpu_depth[0];
    //printf("ker: %i\n", location);

    // checks if this thread needs to calculate
    if (location < dev_totalThreadCount[0])
    {

        SplitNegaMaxSearch(&dev_board[location], &dev_searches[location], depth); 
        //printf("done");
    }
}

//The Search algorithm that calls and manages everything
static int GPUNegaMaxSearch(Chessboard *board, Search *info, int cpu_depth, int gpu_depth)
{

    // Used for when cpuDepth is 2
    int moveCounter[146];
    boards.reserve(pow(100, 3));
    searches.reserve(pow(100, 3)); 

    // initalize the variables to 0
    int totalThreadCount = 0;
    int dcount = 0;
    Movelist saveList[1];
    saveList->moveCount = 0;
    GenerateMoves(board, info, saveList);

    Move validMoves[256];
    Move *validMovePtr = validMoves;

    int *moveCounterPtr = moveCounter;

    // Creates and saves the boards we need to run on the GPU
    CreateNegaMaxSearch(board, info, validMovePtr, moveCounterPtr, cpu_depth, totalThreadCount, dcount);

    // Calculate the block count and total threads per block
    double result = (double)totalThreadCount / (double)256;
    int blockCount = (int)(ceil(result));
    //printf("blockCount: %i", blockCount);
    int threadsPerBlock = 256;
    if (totalThreadCount < threadsPerBlock)
        threadsPerBlock = totalThreadCount;

    //printf("TTC: %i\n", totalThreadCount);

    // Copy arrys over to the device
    Chessboard *boardP;
    Search *searchP;

    boardP = (Chessboard *)malloc(totalThreadCount * sizeof(Chessboard));
    searchP = (Search *)malloc(totalThreadCount * sizeof(Search));

    Chessboard *dev_boards;
    Search *dev_searches;
    int *dev_totalThreadCount;
    int *dev_gpuDepth;

    //allocate space for boards and searchers
    cudaMalloc((void **)&dev_boards, totalThreadCount * sizeof(Chessboard));
    cudaMalloc((void **)&dev_searches, totalThreadCount * sizeof(Search));
    cudaMalloc((void **)&dev_totalThreadCount, sizeof(int));
    cudaMalloc((void **)&dev_gpuDepth, sizeof(int));

    // copy vectors to arrays
    for (int i = 0; i < totalThreadCount; i++)
    {
        boardP[i] = boards[i];
        searchP[i] = searches[i];
    }

    // copy arrays to the GPU
    cudaMemcpy(dev_boards, boardP, totalThreadCount * sizeof(Chessboard), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_searches, searchP, totalThreadCount * sizeof(Search), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_totalThreadCount, &totalThreadCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gpuDepth, &gpu_depth, sizeof(int), cudaMemcpyHostToDevice);

    // CUDA cannot calculate recursive stack size, so we have to set stack size manually
    size_t limit = 2616*(gpu_depth+1); //space taken up by going one deeper, plus the first depth
    cudaDeviceSetLimit(cudaLimitStackSize, limit);

    // for testing
    //threadsPerBlock = 256;

    // create variables and then run our search
    dim3 grid(blockCount, 1, 1);

    kernelSearch<<<grid, threadsPerBlock>>>(dev_boards, dev_searches, dev_totalThreadCount, dev_gpuDepth);
    cudaDeviceSynchronize();

    // copy the data back to the arrays

    cudaMemcpy(boardP, dev_boards, totalThreadCount * sizeof(Chessboard), cudaMemcpyDeviceToHost);
    cudaMemcpy(searchP, dev_searches, totalThreadCount * sizeof(Search), cudaMemcpyDeviceToHost);

    cudaFree(dev_boards);
    cudaFree(dev_searches);

    int retScore = -50000;
    int bestIndex = 0;

    if (cpu_depth == 1)
    {
        // Simple Search
        int reverse = -1;

        for (int i = 0; i < totalThreadCount; i++)
        {
            // printf("Index: %i Move: ", i);
            // PrintMove(searchP[i].bestMove);
            // printf(" Score: %i \n", searchP[i].bestScore);
            if (reverse * searchP[i].bestScore > retScore)
            {
                retScore = reverse * searchP[i].bestScore;
                bestIndex = i;

                //printf("index: %i ", bestIndex);
                //PrintMove(searchP[i].bestMove);
                //printf("\n");
            }
        }

        info->bestMove = validMoves[bestIndex].move;
    }
    else if (cpu_depth == 2)
    {
        // MiniMax adjacent search to determine our best move
        int lastIndex = 0;
        int maxVal = -90000;
        int minVal = 90000;
        for (int i = 0; i < dcount; i++)
        {
            minVal = 90000;
            for (int j = lastIndex; j < moveCounter[i]; j++)
            {
                if (searchP[j].bestScore < minVal)
                {
                    minVal = searchP[j].bestScore;
                }
            }
            if (minVal > maxVal)
            {
                maxVal = minVal;
                bestIndex = i;
            }
            lastIndex = moveCounter[i];
        }
        retScore = maxVal;
        info->bestMove = validMoves[bestIndex].move;
    }
    else
    {
        // This is logically sound as the evalutation function just becomes the array index of that position
        // but will not launch as the number of threads will be too high
        int count = 0;
        retScore = EvalNegaMaxSearch(board, info, searchP, cpu_depth, count);
    }

    // printf("index: %i \n", bestIndex);

    free(boardP);
    free(searchP);

    return retScore;
}

// Function that runs our search and times our functions
static inline void SearchPosition(Chessboard *board, Search *info, int cpuDepth, int gpuDepth)
{

    clock_t start, end;
    start = clock();
    int score;

    if (gpuDepth > 0)
    {
        score = GPUNegaMaxSearch(board, info, cpuDepth, gpuDepth);
    }
    else
    {
        score = RegNegaMaxSearch(board, info, cpuDepth);
    }

    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;

    if (score == 49000)
        return;

    printf("info score cp %d depth %d depth %d\n", score, cpuDepth, gpuDepth);
    printf("Time taken to depth %f seconds\n", time);

    FILE *out_file = fopen("out_file.txt", "a"); // write only
    fprintf(out_file, "%f \n", time);
    fclose(out_file);

    printf("bestmove ");
    PrintMove(info->bestMove);
    printf("\n");

    //printf("Move ordering: %.2f\n",(info->fhf/info->fh));
}


// Function that converts a FEN into a board
void ParseFen(Chessboard *board, char *fen)
{
    ResetBoard(board);

    RankLoop{
        FileLoop{
            int sq = fr2sq(file, rank);

    // parse position
    if (IsOnBoard(sq))
    {
        if (isPieceChar(*fen))
        {
            if (*fen == 'K')
                kingSq(w) = sq;

            else if (*fen == 'k')
                kingSq(b) = sq;

            //printf( "%i", *fen);
            //printf("Here");

            switch (*fen)
            {

            case 114:
                SetSq(sq, 12);
                break; // wR
            case 110:
                SetSq(sq, 10);
                break; // wN
            case 98:
                SetSq(sq, 11);
                break; // wB
            case 113:
                SetSq(sq, 13);
                break; // wQ
            case 112:
                SetSq(sq, 9);
                break; // wP

            case 82:
                SetSq(sq, 4);
                break; //bR
            case 78:
                SetSq(sq, 2);
                break; //bN
            case 66:
                SetSq(sq, 3);
                break; //bB
            case 81:
                SetSq(sq, 5);
                break; //bQ
            case 80:
                SetSq(sq, 1);
                break; //bP
            }

            *fen++;
        }

        if (isDigit(*fen))
        {
            int count = *fen - '0';

            if (!GetSq(sq))
                file++;

            file -= count;
            *fen++;
        }

        if (*fen == '/')
        {
            *fen++;
            file--;
        }
    }
}
}

*fen++;

// parse stats
side = (*fen == 'w') ? w : b;
fen += 2;

while (*fen != ' ')
{
    switch (*fen)
    {
    case 'K':
        castle |= K;
        break;
    case 'Q':
        castle |= Q;
        break;
    case 'k':
        castle |= k;
        break;
    case 'q':
        castle |= q;
        break;

    case '-':
        castle = 0;
    }
    fen++;
}

fen++;

if (*fen != '-')
{
    int file = fen[0] - 'a';
    int rank = fen[1] - '0';
    enPassant = parse2sq(file, rank);
}
}

// Parses a move
int ParseMove(Chessboard *board, Search *info, char *moveStr)
{
    Movelist list[1];
    GenerateMoves(board, info, list);

    int parseFrom = (moveStr[0] - 'a') + (moveStr[1] - '0' - 1) * 16;
    int parseTo = (moveStr[2] - 'a') + (moveStr[3] - '0' - 1) * 16;
    int promPiece = 0;

    int move;

    for (int moveNum = 0; moveNum < list->moveCount; ++moveNum)
    {
        move = list->moves[moveNum].move;

        if (GetMoveSource(move) == parseFrom && GetMoveTarget(move) == parseTo)
        {
            promPiece = GetMovePromPiece(move);

            if (promPiece)
            {
                if ((promPiece == wN || promPiece == bN) && moveStr[4] == 'n')
                    return move;

                else if ((promPiece == wB || promPiece == bB) && moveStr[4] == 'b')
                    return move;

                else if ((promPiece == wR || promPiece == bR) && moveStr[4] == 'r')
                    return move;

                else if ((promPiece == wQ || promPiece == bQ) && moveStr[4] == 'q')
                    return move;

                continue;
            }

            return move;
        }
    }

    return 0;
}

// Performance testing code when "test" is called, Search Position but loops through file for input
static inline void TestSearchPosition(Chessboard *board, Search *info, int cpuDepth, int gpuDepth)
{

    char const *const fileName = "test.txt"; 
    FILE *file = fopen(fileName, "r");
    char line[256];

    while (fgets(line, sizeof(line), file))
    {
        ParseFen(board, line);
        InitSearch(info);
        PrintBoard(board);

        clock_t start, end;
        start = clock();

        int score;
        if (gpuDepth > 0)
        {
            score = GPUNegaMaxSearch(board, info, cpuDepth, gpuDepth);
        }
        else
        {
            score = RegNegaMaxSearch(board, info, cpuDepth);
        }

        end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC;

        if (score == 49000)
            return;

        printf("info score cp %d depth %d depth %d\n", score, cpuDepth, gpuDepth);
        printf("Time taken to depth %f seconds\n", time);

        FILE *out_file = fopen("out_file.txt", "a"); // write only
        fprintf(out_file, "%f \n", time);
        fclose(out_file);

        printf("bestmove ");
        PrintMove(info->bestMove);
        printf("\n");
    }
}

// Hammerhead main
int main(int argc, char *argv[])
{
    // Init everything
    Chessboard board[1];
    Search info[1];
    InitSearch(info);

    char inFen[87];

    int cpuDepth = 2;
    int gpuDepth = 3;

    // If user specifies depth, set depths
    if (argc == 3)
    {
        cpuDepth = atoi(argv[1]);
        gpuDepth = atoi(argv[2]);
    }

    // Larger cpuDepth is not yet implemented so this is a failsafe
    if (cpuDepth > 2){
        cpuDepth = 2;
    }

    // Loop through accepting and analyzing FENs
    while (1)
    {

        printf("Please enter a fen to analyze: \n");

        // take input fen
        fgets(inFen, 87, stdin);

        if (std::strstr(inFen, "quit") || std::strstr(inFen, "done") || std::strstr(inFen, "over"))
        {
            return 0;
        }

        if (std::strstr(inFen, "test"))
        {
            TestSearchPosition(board, info, cpuDepth, gpuDepth);
            return 0;
        }

        ParseFen(board, inFen);

        PrintBoard(board);

        SearchPosition(board, info, cpuDepth, gpuDepth);

    }

    return 0;
}

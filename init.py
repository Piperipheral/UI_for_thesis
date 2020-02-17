import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import pygame
import pygame.surfarray as surfarray
import tensorflow as tf
from keras import backend as K
from keras import callbacks, optimizers
from keras.datasets import mnist
from keras.layers import *
from keras.layers import Add, Dense, Input, Lambda, Layer, Multiply
from keras.models import *
from keras.models import Model, Sequential, load_model
from midiutil.MidiFile import MIDIFile

# ---DATA PREP---
MAX_VECTOR_LENGTH = 30
NOTES = 60
CULL = 0.5
CULL_DELTA = 0.05
MAX_LATENT = 2
# ---MODEL---
LEARNING_RATE = 0.001
LATENT_SPACE_SNAPSHOT = 100
ORIGINAL_SIZE = NOTES * MAX_VECTOR_LENGTH
INTERMIDIATE_SIZE = 200
LATENT_SIZE = 2

# --SOME UI RELATED VARIABLES --
graph_w = 500
graph_h = 500
backgroundMain = (255, 255, 255)
backgroundGraph = (30, 30, 30)
pointCol = (200, 200, 0)
padding = 10

mouse_pressed = False
loop_enable = True

pygame.init()
graphModel = pygame.image.load("graphImg.png")
graph = pygame.display.set_mode(
    (graphModel.get_rect().size[1], graphModel.get_rect().size[1])
)
graph.blit(graphModel, (0, 0))
graph = pygame.transform.scale(graph, (graph_h, graph_w))

surfarraySurface = pygame.display.set_mode((NOTES, MAX_VECTOR_LENGTH))

font = pygame.font.SysFont("monospace", 20)

# Open a window
pygame.font.init()
size = pygame.display.list_modes()[0]
screen = pygame.display.set_mode(size)


pygame.display.set_caption("MusicEdit")
xPoint = 160
yPoint = 160
paramX = 0
paramY = 0

currentMusic = np.zeros((MAX_VECTOR_LENGTH, NOTES))

decoder = Sequential(
    [
        Dense(INTERMIDIATE_SIZE, input_dim=LATENT_SIZE, activation="relu"),
        Dense(ORIGINAL_SIZE, activation="sigmoid"),
    ]
)


def load_the_model():
    decoder.load_weights("theDecoder.h5")


def draw_graph(mouse_pos):
    global xPoint
    global yPoint
    screen.blit(graph, (graph_w + padding, padding))
    pygame.draw.rect(screen, backgroundGraph, (padding, padding, graph_w, graph_h))
    if mouse_pressed:
        xPoint = mouse_pos[0]
        if xPoint > graph_w + padding:
            xPoint = graph_w + padding
        elif xPoint < padding:
            xPoint = padding
        yPoint = mouse_pos[1]
        if yPoint > graph_h + padding:
            yPoint = graph_h + padding
        elif yPoint < padding:
            yPoint = padding
    pygame.draw.circle(screen, pointCol, (xPoint, yPoint), 5)


def render_text(x, y):
    theText = "x: " + str(x) + ", y: " + str(y)
    textsurface = font.render(theText, False, (0, 0, 0))
    screen.blit(textsurface, (padding, graph_h + padding))


def create_music(x, y):
    global surfarraySurface
    global currentMusic
    testPred = [[x, y]]
    ##print(testPred)
    testPred = np.asarray(testPred).reshape(1, LATENT_SIZE)
    test = decoder.predict(testPred).reshape(NOTES, MAX_VECTOR_LENGTH)
    test = np.asarray(test)
    culling_upper = test < CULL
    test[culling_upper] = 0
    test = np.transpose(test)
    currentMusic = test
    test = np.interp(test, (test.min(), test.max()), (0, +100))
    surfarraySurface = surfarray.make_surface(test)
    surfarraySurface = pygame.transform.scale(
        surfarraySurface,
        (int(MAX_VECTOR_LENGTH * 100 / (MAX_VECTOR_LENGTH // 10)), NOTES),
    )


# Main loop
running = True
rand_ix = 0
cur_len = 0
load_the_model()
create_music(0, 0)


def toMidi(inputMan, fname):
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    tickTime = 0.2
    MyMIDI.addTrackName(track, time, "Sample Track")
    MyMIDI.addTempo(track, time, 110)
    the_input = np.zeros((MAX_VECTOR_LENGTH + 1, 127))
    the_input[
        :MAX_VECTOR_LENGTH, ((127 - NOTES) // 2) : ((127 - NOTES) // 2 + NOTES)
    ] = inputMan
    plt.figure(figsize=(10, 10))
    plt.imshow(the_input, cmap="gray", aspect="auto")
    flag = np.zeros((127,))
    startNote = np.zeros((127,))
    myTick = 0
    for xMan in range(len(the_input)):
        myTick += tickTime
        # print("curently at: ", xMan)
        for yMan in range(len(the_input[xMan])):
            if the_input[xMan][yMan] > 0:
                if flag[yMan - 1] == 0:
                    startNote[yMan - 1] = myTick
                flag[yMan - 1] += tickTime
            else:
                if flag[yMan - 1] != 0:
                    pitch = yMan
                    time = startNote[yMan - 1]
                    duration = flag[yMan - 1]
                    volume = 100
                    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
                    """
                    print(
                        "appending: "
                        + str(yMan)
                        + " at "
                        + str(time)
                        + " for "
                        + str(duration)
                    )
                    """
                    flag[yMan - 1] = 0
                    startNote[yMan - 1] = 0

    binfile = open(fname, "wb")
    MyMIDI.writeFile(binfile)
    binfile.close()
    pygame.mixer.music.load(fname)
    pygame.mixer.music.play()


while running:
    # Draw to the screen
    screen.fill(backgroundMain)
    draw_graph(pygame.mouse.get_pos())
    paramX = ((xPoint - padding) / (graph_w / 2) - 1) * MAX_LATENT
    paramY = ((yPoint - padding) / (graph_h / 2) - 1) * MAX_LATENT
    render_text(round(paramX, 2), round(paramY, 2))
    screen.blit(surfarraySurface, (padding, graph_h + padding * 3))

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            if mouse_pressed:
                audio_reset = True
                create_music(paramX, paramY)
            mouse_pressed = 0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = 1
        elif event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                exit()
            if event.key == pygame.K_SPACE:
                toMidi(currentMusic, "theOutput.mid")
            if event.key == pygame.K_TAB:
                pygame.mixer.music.load("theOutput.mid")
                pygame.mixer.music.play()
            if event.key == pygame.K_UP:
                if CULL <= 1:
                    CULL += CULL_DELTA
                    create_music(paramX, paramY)
            if event.key == pygame.K_DOWN:
                if CULL >= 0:
                    CULL -= CULL_DELTA
                    create_music(paramX, paramY)
    pygame.display.flip()
    pygame.time.wait(10)

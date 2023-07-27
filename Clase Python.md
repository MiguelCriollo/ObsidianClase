Lenguaje facil de leer.
Indexasion donde importan los espacios
PyCharm para BackEnd
Funciones y ayuda:

```python
def multiply_by_two(x):

  """Sasadada matamoros"""

  return  x*2

help(multiply_by_two)

def apply_func_twice(func,arg):

  return func(func(arg))

  

def add_five(x):

  return x + 5

print(apply_func_twice(add_five,30))

def greet(person):

  return f"hello, {person}!"

  

def repeat(func, times, arg):

  for _ in range(times):

    print(func(arg))

repeat(greet,3,'OpenAI')
```


Funciones de orden superior:
Retornan una funcion
Booleans y condicionales
```python
a>0 and b>0
```
Listas
```python
fruits = ['apple','banana','cherry']
fruits[1:3]
```
No se puede hacer el fruits[152]='No Se Puede'
```python
respuesta=input("Dame tu signo sodiacal: ")

signo=[["tauro","pispis","cancer","aquario"],["Muerte hoy","muerte manana","no muerte","muere tu perro"]]

print(signo[1][signo[0].index(respuesta)])
```

Bucles
for in recorre objetos

String y diccionarioss

```python
planet = "mars"

num = 4

print(("Planet is {} and num is {}").concatenate(planet,num))
```
```python
planets = [

    'Mercury',

    'Anchoa',

    'Cacho',

    'Hasc',

    'Acf'

]

planet_initials = {

    planet:planet[0]

    for planet in planets

}

print(planet_initials)
```

Librerias externas

python para data science

Boleo loteria 4 numeros y animal
Mismo animal que ganador, puedes pedir otro boleto

```python
import random

  

animales = ['Ballena', 'Tigre', 'Anaconda', 'Panda', 'Tortuga']

  

def randomTicket():

    return [random.randint(0,1) for _ in range(2)]

  

def randomAnimal():

    return animales[random.randint(0,len(animales)-1)]

  

winnerTicket=randomTicket()

winnerAnimal=randomAnimal()

while True:

    ticketUser=randomTicket()

    boleto = input("Si logras adivinar el animal ganador, puedes volver intentar ganar otros premios con nuevo boleto ")

    print (f"Numero ganador es: {winnerTicket} y el animal ganador es:{winnerAnimal}, tu ticket es: {ticketUser}")

    if(winnerTicket==ticketUser and boleto == winnerAnimal):

        print("Ganaste")

        break

    else:

        print("Perdiste")
```
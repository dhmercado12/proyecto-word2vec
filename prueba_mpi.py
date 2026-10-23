from mpi4py import MPI
import socket

comunicador = MPI.COMM_WORLD
rango = comunicador.Get_rank()
total = comunicador.Get_size()
maquina = socket.gethostname()

print(f"Proceso {rango} de {total} ejecutado en {maquina}")

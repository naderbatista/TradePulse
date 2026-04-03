import asyncio
import json
import aiohttp

async def trigger_start():
    session = aiohttp.ClientSession()
    ws = await session.ws_connect("http://localhost:8080/ws")
    # Recebe snapshot inicial
    msg = await ws.receive()
    print("Snapshot recebido")
    # Envia start
    await ws.send_str(json.dumps({"action": "start"}))
    print("Comando 'start' enviado")
    # Aguarda updates
    for _ in range(3):
        msg = await ws.receive()
        data = json.loads(msg.data)
        print(f"  Tipo: {data.get('type')}, Preco: {data.get('current_price', '-')}, Sinal: {data.get('signal', '-')}")
    await ws.close()
    await session.close()

asyncio.run(trigger_start())

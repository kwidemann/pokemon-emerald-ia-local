-- Servidor TCP para mGBA (scripting Lua, mGBA >= 0.10).
-- Protocolo: un comando por línea; una respuesta por línea ("OK <datos>" o "ERR <motivo>").
-- Cargar desde mGBA: Tools > Scripting > File > Load script.
--
-- Comandos:
--   ping
--   frame
--   read8|read16|read32 <addr>        addr en decimal o 0x...
--   readrange <addr> <len>            respuesta en hexadecimal
--   readmulti <addr>:<len> ...        varios rangos en una llamada: OK <frame> <hex> <hex> ...
--   step <mascara_teclas> <frames>    mantiene las teclas N frames y responde al terminar
--   savestate <ruta> | loadstate <ruta>
--   reset                             reinicia la consola (arranca desde el logo)
--   screenshot <ruta>
--
-- La máscara de teclas usa el orden de bits de GBA: A=0 B=1 SELECT=2 START=3
-- RIGHT=4 LEFT=5 UP=6 DOWN=7 R=8 L=9.

local PORT = 8888

local server = nil
local clients = {}  -- id -> socket
local buffers = {}  -- id -> texto recibido sin procesar
local nextId = 1
local pending = nil -- { id = clienteId, untilFrame = n }

local function reply(id, text)
	local sock = clients[id]
	if sock then
		sock:send(text .. "\n")
	end
end

local function dropClient(id)
	local sock = clients[id]
	clients[id] = nil
	buffers[id] = nil
	if pending and pending.id == id then
		pending = nil
	end
	if sock then
		sock:close()
	end
end

local function toHex(data)
	return (data:gsub(".", function(c) return string.format("%02x", string.byte(c)) end))
end

local function handle(id, line)
	local parts = {}
	for token in string.gmatch(line, "%S+") do
		parts[#parts + 1] = token
	end
	local cmd = parts[1]
	if not cmd then
		return
	end
	-- Rutas (pueden tener espacios): todo lo que sigue al comando, sin espacios en los extremos.
	local path = string.match(line, "^%s*%S+%s+(.-)%s*$")

	if cmd == "ping" then
		reply(id, "OK pong")
	elseif cmd == "frame" then
		reply(id, "OK " .. emu:currentFrame())
	elseif cmd == "read8" then
		reply(id, "OK " .. emu:read8(tonumber(parts[2])))
	elseif cmd == "read16" then
		reply(id, "OK " .. emu:read16(tonumber(parts[2])))
	elseif cmd == "read32" then
		reply(id, "OK " .. emu:read32(tonumber(parts[2])))
	elseif cmd == "readrange" then
		reply(id, "OK " .. toHex(emu:readRange(tonumber(parts[2]), tonumber(parts[3]))))
	elseif cmd == "reset" then
		-- Reinicia la consola como si se apagara y encendiera (vuelve al logo/título; no toca el .sav).
		emu:reset()
		reply(id, "OK")
	elseif cmd == "readmulti" then
		-- readmulti <addr>:<len> <addr>:<len> ...  ->  OK <frame> <hex1> <hex2> ...
		local out = { tostring(emu:currentFrame()) }
		for i = 2, #parts do
			local addr, len = string.match(parts[i], "^(%w+):(%d+)$")
			if not addr then
				reply(id, "ERR rango invalido: " .. parts[i])
				return
			end
			out[#out + 1] = toHex(emu:readRange(tonumber(addr), tonumber(len)))
		end
		reply(id, "OK " .. table.concat(out, " "))
	elseif cmd == "step" then
		if pending then
			reply(id, "ERR step en curso")
			return
		end
		local mask = tonumber(parts[2]) or 0
		local frames = tonumber(parts[3]) or 1
		emu:setKeys(mask)
		pending = { id = id, untilFrame = emu:currentFrame() + frames }
	elseif cmd == "savestate" then
		reply(id, (path and emu:saveStateFile(path)) and "OK" or "ERR savestate")
	elseif cmd == "loadstate" then
		reply(id, (path and emu:loadStateFile(path)) and "OK" or "ERR loadstate")
	elseif cmd == "screenshot" then
		if not path then
			reply(id, "ERR falta la ruta")
			return
		end
		emu:screenshot(path)
		reply(id, "OK")
	else
		reply(id, "ERR comando desconocido: " .. cmd)
	end
end

local function onReceived(id)
	local sock = clients[id]
	if not sock then
		return
	end
	local data, err = sock:receive(4096)
	if not data then
		if err ~= socket.ERRORS.AGAIN then
			dropClient(id)
		end
		return
	end
	local buf = (buffers[id] or "") .. data
	while true do
		local nl = string.find(buf, "\n", 1, true)
		if not nl then
			break
		end
		local line = string.sub(buf, 1, nl - 1)
		buf = string.sub(buf, nl + 1)
		handle(id, line)
	end
	buffers[id] = buf
end

local function onAccept()
	local sock = server:accept()
	if not sock then
		return
	end
	local id = nextId
	nextId = nextId + 1
	clients[id] = sock
	buffers[id] = ""
	sock:add("received", function() onReceived(id) end)
	sock:add("error", function() dropClient(id) end)
	console:log("pokemon-ia: cliente " .. id .. " conectado")
end

-- Al completar los frames pedidos por `step`, se responde con el frame actual.
callbacks:add("frame", function()
	if pending and emu:currentFrame() >= pending.untilFrame then
		local id = pending.id
		pending = nil
		reply(id, "OK " .. emu:currentFrame())
	end
end)

local err
server, err = socket.bind(nil, PORT)
if not server then
	console:error("pokemon-ia: no se pudo abrir el puerto " .. PORT .. ": " .. tostring(err))
else
	server:add("received", onAccept)
	local ok
	ok, err = server:listen()
	if ok == nil then
		console:error("pokemon-ia: listen falló: " .. tostring(err))
	else
		console:log("pokemon-ia: escuchando en el puerto " .. PORT)
	end
end

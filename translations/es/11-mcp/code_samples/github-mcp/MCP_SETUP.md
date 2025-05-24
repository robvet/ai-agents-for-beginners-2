<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:15:45+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "es"
}
-->
# Guía de Integración del Servidor MCP

## Requisitos Previos
- Node.js instalado (versión 14 o superior)
- Gestor de paquetes npm
- Entorno de Python con las dependencias necesarias

## Pasos de Configuración

1. **Instalar el Paquete del Servidor MCP**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **Iniciar el Servidor MCP**  
   ```bash
   npx @modelcontextprotocol/server-github
   ```  
   El servidor debería iniciarse y mostrar una URL de conexión.

3. **Verificar la Conexión**  
   - Busca el ícono de enchufe (🔌) en tu interfaz de Chainlit  
   - Debería aparecer un número (1) junto al ícono de enchufe indicando conexión exitosa  
   - La consola debería mostrar: "GitHub plugin setup completed successfully" (junto con líneas adicionales de estado)

## Solución de Problemas

### Problemas Comunes

1. **Conflicto de Puerto**  
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```  
   Solución: Cambia el puerto usando:  
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **Problemas de Autenticación**  
   - Asegúrate de que las credenciales de GitHub estén configuradas correctamente  
   - Verifica que el archivo .env contenga los tokens requeridos  
   - Confirma el acceso a la API de GitHub

3. **Fallo de Conexión**  
   - Confirma que el servidor esté corriendo en el puerto esperado  
   - Revisa la configuración del firewall  
   - Verifica que el entorno de Python tenga los paquetes necesarios

## Verificación de Conexión

Tu servidor MCP está correctamente conectado cuando:  
1. La consola muestra "GitHub plugin setup completed successfully"  
2. Los registros de conexión muestran "✓ MCP Connection Status: Active"  
3. Los comandos de GitHub funcionan en la interfaz de chat

## Variables de Entorno

Requeridas en tu archivo .env:  
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## Prueba de Conexión

Envía este mensaje de prueba en el chat:  
```
Show me the repositories for username: [GitHub Username]
```  
Una respuesta exitosa mostrará información del repositorio.

**Aviso Legal**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables por malentendidos o interpretaciones erróneas derivadas del uso de esta traducción.
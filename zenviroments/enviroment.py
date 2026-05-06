from environment_obstaculos_lidar import SpiderEnv as SpiderEnvObstaculosLidar
from environment_obstaculos_sin_lidar import SpiderEnv as SpiderEnvObstaculosSinLidar
from environment_sin_obstaculos import SpiderEnv as SpiderEnvSinObstaculos

ENV_VARIANTS = {
    "sin_obstaculos": SpiderEnvSinObstaculos,
    "obstaculos_sin_lidar": SpiderEnvObstaculosSinLidar,
    "obstaculos_lidar": SpiderEnvObstaculosLidar,
}


def get_env_class(name: str):
    key = name.lower()
    if key not in ENV_VARIANTS:
        valid = ", ".join(sorted(ENV_VARIANTS))
        raise ValueError(f"Entorno '{name}' desconocido. Opciones: {valid}")
    return ENV_VARIANTS[key]


# Export por defecto para mantener compatibilidad con imports existentes.
SpiderEnv = SpiderEnvSinObstaculos

__all__ = ["SpiderEnv", "get_env_class", "ENV_VARIANTS"]

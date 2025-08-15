# docstringer — Автоматический генератор docstring для Python/Django/DRF

**docstringer** — это инструмент для автоматической генерации докстрингов (docstrings) в Python-проектах, ориентированный на проекты Django/DRF. Использует Google Gemini API для генерации лаконичных и информативных докстрингов в стиле PEP 257 на русском языке.

---

## Возможности

- Автоматически находит все Python-файлы в проекте (с учётом исключений)
- Анализирует модули, классы, функции и методы без докстрингов
- Генерирует докстринги с помощью Google Gemini (AI)
- Вставляет докстринги непосредственно в код (или показывает их в режиме dry-run)
- Гибкая настройка через pyproject.toml
- Поддержка подробного вывода (verbose)
- Удобный CLI-интерфейс с прогресс-баром и цветным выводом (rich)

---

## Требования

- Python 3.8+
- Google Gemini API ключ ([как получить](https://ai.google.dev/))
- Установленные зависимости:
  - `google-generativeai`
  - `rich`
  - `python-dotenv`
  - `toml` (для Python < 3.11)

---

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/docstringer.git
   cd docstringer
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   # или вручную:
   pip install google-generativeai rich python-dotenv toml
   ```

---

## Быстрый старт

1. Получите API-ключ Gemini и добавьте его в переменные окружения:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   # или создайте .env файл с содержимым:
   # GEMINI_API_KEY=your-gemini-api-key
   ```
2. Запустите генератор для вашего проекта:
   ```bash
   python auto_docstringer.py /path/to/your/project
   ```

---

## Примеры использования

- **Анализ и генерация докстрингов с изменением файлов:**
  ```bash
  python auto_docstringer.py /path/to/project
  ```
- **Только показать, что будет сгенерировано (dry-run):**
  ```bash
  python auto_docstringer.py /path/to/project --dry-run
  ```
- **Передать API-ключ явно:**
  ```bash
  python auto_docstringer.py /path/to/project --api-key=YOUR_KEY
  ```
- **Включить подробный вывод:**
  ```bash
  python auto_docstringer.py /path/to/project --verbose
  ```
- **Настроить количество повторных попыток и задержку:**
  ```bash
  python auto_docstringer.py /path/to/project --max-retries=5 --retry-delay=60
  ```

---

## Конфигурация через pyproject.toml

Можно задать параметры в секции `[tool.auto_docstringer]` вашего `pyproject.toml`:

```toml
[tool.auto_docstringer]
exclude = ["*/migrations/*", "manage.py", "tests/*", "**/__init__.py"]
model_name = "gemini-1.5-flash"
```

- **exclude** — список паттернов для исключения файлов/директорий
- **model_name** — имя модели Gemini (по умолчанию: gemini-1.5-flash)

---

## Переменные окружения

- `GEMINI_API_KEY` — API-ключ Gemini (можно передать через аргумент или .env)

---

## Принцип работы

1. Находит все Python-файлы в проекте (с учётом исключений)
2. Для каждого файла строит AST и ищет модули, классы, функции без докстрингов
3. Для каждого такого узла отправляет фрагмент кода в Gemini и получает докстринг
4. Вставляет сгенерированный докстринг в исходный код (или показывает его в dry-run)
5. Выводит подробную статистику по завершении работы

---

## Возможные ошибки и их решение

- **Нет API-ключа:**
  - Проверьте переменную окружения `GEMINI_API_KEY` или используйте `--api-key`
- **Превышена квота Gemini API:**
  - Используйте параметры `--max-retries` и `--retry-delay` для повторных попыток
- **Не установлены зависимости:**
  - Установите их через `pip install google-generativeai rich python-dotenv toml`
- **Ошибка синтаксиса в файле:**
  - Исправьте ошибки в исходном коде Python-файлов

---

## Лицензия

[MIT License](LICENSE)

---

## Автор

- [3ShishkaN3](https://github.com/3ShishkaN3)

---

**Pull requests и предложения приветствуются!**

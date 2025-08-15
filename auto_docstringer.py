#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import fnmatch
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import toml as tomllib
    except ModuleNotFoundError:
        print("Ошибка: Не найдены библиотеки 'tomllib' (Python 3.11+) или 'toml'.")
        print("Установите 'toml': pip install toml")
        sys.exit(1)

try:
    import google.generativeai as genai
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.syntax import Syntax
except ModuleNotFoundError:
    print("Ошибка: Не найдены необходимые библиотеки 'google-generativeai' или 'rich'.")
    print("Установите их: pip install google-generativeai rich")
    sys.exit(1)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CONFIG_SECTION = "tool.auto_docstringer"
DEFAULT_EXCLUDE_PATTERNS = ["*/migrations/*", "manage.py", "tests/*", "**/__init__.py"]
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
API_KEY_ENV_VAR = "GEMINI_API_KEY"

console = Console()


def load_config(project_path: Path) -> Dict[str, Any]:
    """Загружает конфигурацию из pyproject.toml."""
    config_file = project_path / "pyproject.toml"
    if not config_file.is_file():
        log.warning(f"Файл конфигурации {config_file} не найден.")
        return {}
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        config_parts = CONFIG_SECTION.split(".")
        current = data
        for part in config_parts:
            if part not in current:
                log.warning(
                    f"Секция '{part}' не найдена в конфигурации. Полный путь: {CONFIG_SECTION}"
                )
                return {}
            current = current[part]

        log.debug(f"Загруженная конфигурация: {current}")
        return current
    except Exception as e:
        log.error(f"Ошибка чтения {config_file}: {e}")
        return {}


def find_python_files(project_path: Path, exclude_patterns: List[str]) -> List[Path]:
    """Находит все Python файлы в проекте, исключая указанные паттерны и сам скрипт."""
    py_files = []
    project_root_str = str(project_path.resolve())
    try:
        script_path = Path(__file__).resolve()
    except NameError:
        script_path = None
        log.warning(
            "Не удалось определить путь к запущенному скрипту, самоисключение может не сработать."
        )

    processed_patterns = []
    for pattern in exclude_patterns:
        if pattern.endswith("/"):
            base_pattern = pattern.rstrip("/")
            processed_patterns.append(f"{base_pattern}/**")
            processed_patterns.append(base_pattern)
            processed_patterns.append(f"{base_pattern}*")
        else:
            processed_patterns.append(pattern)

    root_dirs_to_exclude = [
        pattern.rstrip("/")
        for pattern in exclude_patterns
        if pattern.endswith("/") and "/" not in pattern.rstrip("/")
    ]

    log.debug(f"Используемые паттерны исключения (обработанные): {processed_patterns}")

    for path in project_path.rglob("*.py"):
        log.debug(f"Проверка файла: {path}")

        if script_path and path.resolve() == script_path:
            log.debug(f"Исключен сам скрипт: {path.relative_to(project_path)}")
            continue

        relative_path = path.relative_to(project_path)
        normalized_path = relative_path.as_posix()

        log.debug(f"  Нормализованный путь: {normalized_path}")

        is_excluded = False

        for root_dir in root_dirs_to_exclude:
            if normalized_path.startswith(f"{root_dir}/"):
                log.debug(f"    -> ИСКЛЮЧЕНО (корневая директория '{root_dir}/')")
                is_excluded = True
                break

        if not is_excluded:
            for pattern in processed_patterns:
                log.debug(f"    Проверка паттерна: '{pattern}'")

                if fnmatch.fnmatch(normalized_path, pattern):
                    is_excluded = True
                    log.debug(f"      -> ИСКЛЮЧЕНО (паттерн '{pattern}')")
                    break

        if not is_excluded:
            log.debug(f"  -> ВКЛЮЧЕН файл: {normalized_path}")
            py_files.append(path)

    return py_files


class DocstringFinder(ast.NodeVisitor):
    """
    Обходит AST и находит узлы (модуль, классы, функции) без докстрингов.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.nodes_without_docstrings: List[Tuple[ast.AST, str]] = []

    def _get_source_segment(self, node: ast.AST) -> Optional[str]:
        """Пытается получить исходный код для узла."""
        try:
            if sys.version_info >= (3, 8):
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    return ast.get_source_segment(source, node)
                except Exception:
                    log.warning(
                        f"Не удалось получить сегмент кода для узла в {self.file_path}:{node.lineno}"
                    )
                    return None
            else:
                log.warning(
                    "ast.get_source_segment недоступен в этой версии Python. Генерация может быть менее точной."
                )
                return None
        except Exception as e:
            log_line_info = f":{node.lineno}" if hasattr(node, "lineno") else ":module"
            log.debug(
                f"Генерация докстринга для {node_type} '{name}' в {relative_file_path}{log_line_info}"
            )
            return None

    def visit_Module(self, node: ast.Module):
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            self.nodes_without_docstrings.append(
                (node, f"# Module: {self.file_path.name}")
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name.startswith("_") and not (
            node.name.startswith("get_")
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
        ):
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                segment = self._get_source_segment(node)
                if segment:
                    self.nodes_without_docstrings.append((node, segment))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not node.name.startswith("_"):
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                segment = self._get_source_segment(node)
                if segment:
                    self.nodes_without_docstrings.append((node, segment))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if not node.name.startswith("_"):
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                segment = self._get_source_segment(node)
                if segment:
                    self.nodes_without_docstrings.append((node, segment))
        self.generic_visit(node)


def generate_docstring_with_gemini(
    code_snippet: str,
    node_type: str,
    model: genai.GenerativeModel,
    max_retries: int = 3,
    retry_delay: int = 35,
) -> Optional[str]:
    """
    Генерирует докстринг для фрагмента кода с помощью Gemini.

    При превышении квоты API добавляет задержку и повторяет попытку до max_retries раз.

    Args:
        code_snippet: Фрагмент кода для генерации докстринга
        node_type: Тип узла AST (Module, FunctionDef, ClassDef и т.д.)
        model: Экземпляр модели Gemini
        max_retries: Максимальное количество повторных попыток при ошибке квоты
        retry_delay: Время ожидания в секундах между попытками

    Returns:
        Сгенерированный докстринг или None в случае ошибки
    """
    prompt = f"""
    Проанализируй следующий фрагмент кода Python ({node_type}) из проекта Django/DRF и сгенерируй для него лаконичный и информативный докстринг в стиле PEP 257 на русском языке.
    Включи краткое описание того, что делает код, его аргументы (если это функция/метод) с их типами (если очевидно) и что он возвращает (если применимо).
    Не добавляй никакого другого текста, кроме самого докстринга в тройных кавычках (`\"\"\"Docstring goes here.\"\"\"`).
    Не используй markdown-блоки кода (```python ... ```) вокруг докстринга.

    Фрагмент кода:
    ```python
    {code_snippet}
    ```

    Сгенерируй только сам докстринг:
    """

    retry_count = 0
    while retry_count <= max_retries:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.4,
                # max_output_tokens=200
            )
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if not response.parts:
                log.warning(
                    f"Gemini вернул пустой ответ для фрагмента:\n{code_snippet[:100]}..."
                )
                return None

            generated_text = response.text.strip()

            if generated_text.startswith('"""') and generated_text.endswith('"""'):
                docstring = generated_text[3:-3].strip()
                if docstring.startswith("```python"):
                    docstring = docstring[len("```python") :].strip()
                if docstring.endswith("```"):
                    docstring = docstring[: -len("```")].strip()
                return f'"""{docstring}"""'

            elif generated_text.startswith("'''") and generated_text.endswith("'''"):
                docstring = generated_text[3:-3].strip()
                if docstring.startswith("```python"):
                    docstring = docstring[len("```python") :].strip()
                if docstring.endswith("```"):
                    docstring = docstring[: -len("```")].strip()
                return f"'''{docstring}'''"

            else:
                log.warning(
                    f"Gemini вернул текст не в формате докстринга:\n{generated_text}"
                )
                if len(generated_text.splitlines()) > 1:
                    return f'"""{generated_text}"""'
                return None

        except Exception as e:
            if "quota" in str(e).lower() and retry_count < max_retries:
                wait_time = retry_delay
                if "retry_delay" in str(e):
                    try:
                        import re

                        retry_info = re.search(
                            r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e)
                        )
                        if retry_info:
                            wait_time = int(retry_info.group(1)) + 3
                    except Exception:
                        pass

                retry_count += 1
                log.warning(
                    f"Превышена квота Gemini API. Ожидание {wait_time} сек перед повторной попыткой ({retry_count}/{max_retries})..."
                )
                import time

                time.sleep(wait_time)
                continue
            else:
                log.error(f"Ошибка при вызове Gemini API: {e}")
                if "API key not valid" in str(e):
                    console.print("[bold red]Ошибка: Неверный API ключ Gemini.[/]")
                elif "quota" in str(e).lower():
                    console.print(
                        "[bold yellow]Превышена квота Gemini API. Достигнут лимит повторных попыток.[/]"
                    )
                return None

    return None


def add_docstrings_to_file(
    file_path: Path, changes: List[Tuple[ast.AST, str]], dry_run: bool
):
    """Добавляет сгенерированные докстринги в файл."""
    if not changes:
        return

    if dry_run:
        console.print(
            f"[cyan]Обнаружены места для {len(changes)} докстрингов в:[/cyan] {file_path.relative_to(Path.cwd())}"
        )
        for node, docstring in changes:
            node_type = type(node).__name__
            name = getattr(node, "name", "module")

            if hasattr(node, "lineno"):
                console.print(f"  - {node_type} '{name}' (строка {node.lineno})")
            else:
                console.print(f"  - {node_type} '{name}'")

            syntax = Syntax(docstring, "python", theme="default", line_numbers=False)
            console.print(syntax)
            console.print("-" * 20)
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        changes.sort(key=lambda item: getattr(item[0], "lineno", 0), reverse=True)

        modified = False
        for node, docstring in changes:
            line_index = -1
            indentation = ""

            if isinstance(node, ast.Module):
                insert_pos = 0
                if lines:
                    if lines[0].startswith("#!"):
                        insert_pos = 1
                    if len(lines) > insert_pos and lines[insert_pos].strip().startswith(
                        "# -*- coding:"
                    ):
                        insert_pos += 1
                while insert_pos < len(lines) and not lines[insert_pos].strip():
                    lines.pop(insert_pos)

                docstring_lines = [line + "\n" for line in docstring.splitlines()]
                if docstring_lines:
                    lines[insert_pos:insert_pos] = docstring_lines + ["\n"]
                    modified = True

            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                line_index = node.lineno - 1
                if line_index < len(lines):
                    if node.body:
                        first_body_line_index = node.body[0].lineno - 1
                        if first_body_line_index < len(lines):
                            leading_whitespace = len(
                                lines[first_body_line_index]
                            ) - len(lines[first_body_line_index].lstrip(" "))
                            indentation = " " * leading_whitespace
                        else:
                            indentation = " " * (node.col_offset + 4)
                    else:
                        indentation = " " * (node.col_offset + 4)

                    docstring_lines = textwrap.indent(
                        docstring, indentation
                    ).splitlines()
                    docstring_lines = [line + "\n" for line in docstring_lines]

                    if docstring_lines:
                        lines[line_index + 1 : line_index + 1] = docstring_lines
                        modified = True
                else:
                    log.warning(
                        f"Некорректный номер строки {node.lineno} для узла в {file_path}"
                    )

            else:
                log.warning(
                    f"Неподдерживаемый тип узла для вставки докстринга: {type(node).__name__}"
                )

        if modified and not dry_run:
            console.print(
                f"[green]Обновлен файл:[/green] {file_path.relative_to(Path.cwd())}"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    except FileNotFoundError:
        console.print(
            f"[bold red]Ошибка: Файл не найден при попытке записи:[/bold red] {file_path}"
        )
    except Exception as e:
        console.print(
            f"[bold red]Ошибка при модификации файла {file_path}:[/bold red] {e}"
        )
        log.exception(f"Подробности ошибки для {file_path}:")


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическая генерация Python docstrings для Django/DRF проекта с использованием Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project_path", type=str, help="Путь к корневой директории Django/DRF проекта."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(API_KEY_ENV_VAR),
        help=f"API ключ Gemini. По умолчанию используется переменная окружения {API_KEY_ENV_VAR}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Выполнить анализ и показать, какие докстринги будут сгенерированы, но не изменять файлы.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Включить подробный вывод (уровень DEBUG).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Максимальное количество повторных попыток при превышении квоты API (по умолчанию: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=35,
        help="Время ожидания в секундах между повторными попытками (по умолчанию: 35).",
    )

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if not args.api_key:
        console.print(f"[bold red]Ошибка: API ключ Gemini не найден.[/]")
        console.print(
            f"Установите переменную окружения {API_KEY_ENV_VAR} или передайте ключ через аргумент --api-key."
        )
        sys.exit(1)

    project_path = Path(args.project_path).resolve()
    if not project_path.is_dir():
        console.print(
            f"[bold red]Ошибка: Указанный путь '{args.project_path}' не является директорией или не существует.[/]"
        )
        sys.exit(1)

    config = load_config(project_path)
    exclude_patterns = config.get("exclude", DEFAULT_EXCLUDE_PATTERNS)
    gemini_model_name = config.get("model_name", DEFAULT_GEMINI_MODEL)

    console.print(
        Panel(
            f"Запуск генератора докстрингов\n"
            f"Проект: [cyan]{project_path}[/cyan]\n"
            f"Модель Gemini: [cyan]{gemini_model_name}[/cyan]\n"
            f"Макс. повторных попыток: [cyan]{args.max_retries}[/cyan]\n"
            f"Задержка между попытками: [cyan]{args.retry_delay} сек[/cyan]\n"
            f"Режим 'dry run': {'[bold green]Включен[/]' if args.dry_run else '[bold red]Выключен[/]'}",
            title="[bold blue]Настройки[/]",
            border_style="blue",
        )
    )

    console.print("\n[blue]Поиск Python файлов...[/]")
    py_files = find_python_files(project_path, exclude_patterns)

    if not py_files:
        console.print(
            "[yellow]Не найдено ни одного Python файла для обработки (с учетом исключений).[/]"
        )
        sys.exit(0)

    console.print(f"Найдено {len(py_files)} файлов для анализа:")
    # for i, file in enumerate(py_files):
    #     console.print(f"  {i+1}. {file.relative_to(project_path)}")
    from rich.filesize import decimal
    from rich.markup import escape
    from rich.text import Text
    from rich.tree import Tree

    tree = Tree(
        f":open_file_folder: [link file://{project_path}]{escape(project_path.name)}",
        guide_style="bold bright_blue",
    )

    paths_dict = {}
    for file_path in py_files:
        parts = file_path.relative_to(project_path).parts
        current_level = paths_dict
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current_level.setdefault("_files", []).append(file_path)
            else:
                current_level = current_level.setdefault(part, {})

    def build_tree(node, current_dict):
        dirs = sorted([d for d in current_dict if d != "_files"])
        files = sorted(current_dict.get("_files", []), key=lambda p: p.name)

        for d in dirs:
            branch = node.add(
                f":file_folder: [bold magenta]{escape(d)}[/]", guide_style="magenta"
            )
            build_tree(branch, current_dict[d])
        for f in files:
            text_filename = Text(f.name, "green")
            text_filename.highlight_regex(r"\.(py)$", "bold green")
            text_filename.stylize(f"link file://{f}")
            icon = "🐍 "
            node.add(Text(icon) + text_filename)

    build_tree(tree, paths_dict)
    console.print(tree)
    console.print("-" * 60)

    try:
        genai.configure(api_key=args.api_key)
        model = genai.GenerativeModel(gemini_model_name)
        model.generate_content(
            "Test", generation_config=genai.types.GenerationConfig(max_output_tokens=5)
        )
    except Exception as e:
        console.print(
            f"[bold red]Ошибка инициализации Gemini ({gemini_model_name}): {e}[/]"
        )
        sys.exit(1)

    console.print(
        f"\n[blue]Анализ файлов и генерация докстрингов ({'dry run' if args.dry_run else 'применение изменений'})...[/]"
    )

    files_processed = 0
    files_updated = 0
    docstrings_generated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[green]Обработка файлов...", total=len(py_files))

        for file_path in py_files:
            relative_file_path = file_path.relative_to(project_path)
            progress.update(task, description=f"[green]Файл: {relative_file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                tree = ast.parse(source_code, filename=str(file_path))

                finder = DocstringFinder(file_path)
                finder.visit(tree)

                nodes_to_document = finder.nodes_without_docstrings
                changes_for_file: List[Tuple[ast.AST, str]] = (
                    []
                )  # (node, generated_docstring)

                if nodes_to_document:
                    progress.update(
                        task,
                        description=f"[yellow]Генерация для: {relative_file_path} ({len(nodes_to_document)} шт.)",
                    )
                    for node, code_segment in nodes_to_document:
                        node_type = type(node).__name__
                        name = getattr(node, "name", "module")
                        log_line_info = (
                            f":{node.lineno}" if hasattr(node, "lineno") else ":module"
                        )
                        log.debug(
                            f"Генерация докстринга для {node_type} '{name}' в {relative_file_path}{log_line_info}"
                        )

                        context_code = (
                            code_segment if code_segment else f"# {node_type}: {name}"
                        )
                        max_context_len = 2000
                        if len(context_code) > max_context_len:
                            context_code = (
                                context_code[:max_context_len] + "\n# ... (код урезан)"
                            )
                            log.warning(
                                f"Код для {node_type} '{name}' в {relative_file_path} урезан до {max_context_len} символов."
                            )

                        generated_docstring = generate_docstring_with_gemini(
                            context_code,
                            node_type,
                            model,
                            max_retries=args.max_retries,
                            retry_delay=args.retry_delay,
                        )

                        if generated_docstring:
                            log.debug(
                                f"Сгенерирован докстринг для {node_type} '{name}':\n{generated_docstring}"
                            )
                            changes_for_file.append((node, generated_docstring))
                            docstrings_generated += 1
                        else:
                            log.warning(
                                f"Не удалось сгенерировать докстринг для {node_type} '{name}' в {relative_file_path}"
                            )
                            progress.update(
                                task,
                                description=f"[red]Ошибка генерации: {relative_file_path}",
                            )

                    if changes_for_file:
                        add_docstrings_to_file(
                            file_path, changes_for_file, args.dry_run
                        )
                        if not args.dry_run:
                            files_updated += 1

                files_processed += 1

            except SyntaxError as e:
                console.print(
                    f"[bold red]Ошибка синтаксиса в файле {relative_file_path}: {e}[/]"
                )
                progress.update(
                    task, description=f"[red]Ошибка синтаксиса: {relative_file_path}"
                )
            except FileNotFoundError:
                console.print(
                    f"[bold red]Ошибка: Файл не найден при чтении:[/bold red] {relative_file_path}"
                )
                progress.update(
                    task, description=f"[red]Файл не найден: {relative_file_path}"
                )
            except Exception as e:
                console.print(
                    f"[bold red]Неизвестная ошибка при обработке файла {relative_file_path}: {e}[/]"
                )
                log.exception(f"Подробности ошибки для {relative_file_path}:")
                progress.update(
                    task, description=f"[red]Ошибка обработки: {relative_file_path}"
                )

            progress.advance(task)

    console.print("\n" + "=" * 60)
    console.print("[bold blue]Завершено.[/]")
    console.print(f"Всего файлов проанализировано: {files_processed}")
    if args.dry_run:
        console.print(
            f"Потенциально будет сгенерировано докстрингов: {docstrings_generated}"
        )
        console.print("[yellow]Режим 'dry run': Файлы не были изменены.[/]")
    else:
        console.print(
            f"Всего докстрингов сгенерировано и добавлено: {docstrings_generated}"
        )
        console.print(f"Файлов обновлено: {files_updated}")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

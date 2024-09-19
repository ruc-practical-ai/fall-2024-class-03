echo "Installing http-server..."

npm i -g http-server

if [ $? -eq 0 ]; then
    echo "http-server installed!"
else
    echo "Failed to install http-server."
    exit 1
fi

echo "Installing Poetry..."

pip install poetry

if [ $? -eq 0 ]; then
    echo "Poetry installed!"
else
    echo "Failed to install Poetry."
    exit 1
fi

echo "Configuring Poetry virtual environments..."

poetry config virtualenvs.in-project true

if [ $? -eq 0 ]; then
    echo "Virtual environments configured!"
else
    echo "Failed to configure virtual environments."
    exit 1
fi

echo "Installing repository..."

poetry install --with dev --no-root

if [ $? -eq 0 ]; then
    echo "Repository dependencies installed!"
else
    echo "Failed to install repository dependencies."
    exit 1
fi

echo "Success!"

exit 0
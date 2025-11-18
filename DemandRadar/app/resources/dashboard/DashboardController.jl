module DashboardController

using Genie.Renderer.Html

function index()
  html(:dashboard, :index)
end

end